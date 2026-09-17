from .setup import ModelSetup, DataSetup, OptimizerSetup
from .loss import LossFunctions
from .training_loop import TrainingLoop
from .checkpoint import CheckpointManager, WeightAnomalyDetector


def is_main_process(rank):
    return rank == 0

class Trainer:
    def __init__(self, config):
        self.config = config
        self.seed = ModelSetup.setup_reproducibility(config)
        self.device, self.gpu_ids, self.is_multi_gpu, self.rank = ModelSetup.setup_gpu(config)
        self.use_amp, self.scaler = ModelSetup.setup_amp(config, self.rank)
        model_without_ddp = ModelSetup.create_model(config, self.device, self.rank)
        self.model = ModelSetup.setup_multi_gpu(model_without_ddp, self.gpu_ids, self.is_multi_gpu)
        if is_main_process(self.rank):
            ModelSetup.print_model_info(self.model)
            print(f"Reproducibility seed: {self.seed}")
        self.train_loader, self.val_loaders, self.train_dataset = DataSetup.setup_data_loaders(config, self.rank)
        self.split_manifest = getattr(self.train_dataset, "split_manifest", None)
        self.loss_functions = LossFunctions(config, self.device)
        self.optimizer, self.scheduler = OptimizerSetup.setup_optimizer(
            self.model, config, self.rank
        )
        self.training_loop = TrainingLoop(
            self.model,
            self.loss_functions,
            self.device,
            self.use_amp,
            config,
            scaler=self.scaler,
        )

        training_cfg = self.config.get('training', {})
        checkpoint_dir = training_cfg.get('checkpoint_dir', 'checkpoints')
        self.checkpoint_manager = CheckpointManager(
            self.model,
            self.is_multi_gpu,
            checkpoint_dir=checkpoint_dir,
            config=self.config,
            split_manifest=self.split_manifest,
        )
        configured_monitor = str(training_cfg.get('checkpoint_monitor', 'clean')).lower()
        self.monitor_val_name = configured_monitor if configured_monitor in self.val_loaders else next(iter(self.val_loaders.keys()))
        self.train_losses = []
        self.val_losses = []
        self.val_loss_history_by_loader = {name: [] for name in self.val_loaders.keys()}
        self.best_val_loss = float('inf')
        self.current_epoch = 0

    @staticmethod
    def _format_dae_terms(terms):
        return (
            f"Total: {terms.get('total', float('nan')):.6f}, "
            f"Pos: {terms.get('recon_raw', float('nan')):.6f}, "
            f"GroupSpeed: {terms.get('group_speed_recon_raw', float('nan')):.6f}, "
            f"Pair: {terms.get('pair_consistency_raw', float('nan')):.6f}, "
            f"Group: {terms.get('group_consistency_raw', float('nan')):.6f}"
        )

    def train(self):
        training_config = self.config.get('training', {})
        validation_interval = training_config.get('validation_interval', 1)
        if is_main_process(self.rank):
            print("Starting training with denoising autoencoder loss...")
            print(f"Total epochs: {training_config.get('epochs', 200)}")
            print(f"Learning rate: {training_config.get('learning_rate', 1e-4)}")
            print(f"Batch size: {training_config.get('batch_size', 64)}")

        for epoch in range(training_config.get('epochs', 200)):
            self.current_epoch = epoch

            if hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(epoch)
            for val_loader in self.val_loaders.values():
                if hasattr(val_loader.dataset, 'set_epoch'):
                    val_loader.dataset.set_epoch(epoch)
            if is_main_process(self.rank):
                base_dataset = getattr(self.train_loader.dataset, "base_dataset", None)
                augmentation = getattr(base_dataset, "augmentation1", None)
                if augmentation is not None:
                    print(
                        f'[AUG] epoch={epoch}, applied augmentation: '
                        f'{augmentation.get_current_aug_params()}'
                    )
            if is_main_process(self.rank):
                print(f"\nEpoch {epoch+1}/{training_config.get('epochs', 200)}")
                print("-" * 50)
            train_loss, train_cont = self.training_loop.train_epoch(
                self.train_loader, self.optimizer, current_epoch=epoch
            )
            self.scheduler.step()
            self.train_losses.append(train_loss)
            monitor_val_loss = None
            val_results = {}
            if (epoch + 1) % validation_interval == 0 or epoch == training_config.get('epochs', 200) - 1:
                for loader_name, val_loader in self.val_loaders.items():
                    split_loss, split_terms = self.training_loop.validate_epoch(
                        val_loader,
                        epoch=epoch + 1,
                        split_name=loader_name,
                    )
                    val_results[loader_name] = (split_loss, split_terms)
                    self.val_loss_history_by_loader[loader_name].append(split_loss)
                monitor_val_loss, _ = val_results[self.monitor_val_name]
                self.val_losses.append(monitor_val_loss)
                if is_main_process(self.rank):
                    print(
                        "Validation monitor - "
                        f"{self.monitor_val_name}: {monitor_val_loss:.6f}"
                    )
            else:
                for loader_name in self.val_loaders.keys():
                    self.val_loss_history_by_loader[loader_name].append(None)
                self.val_losses.append(None)
            if is_main_process(self.rank):
                print(f"Train Loss: {train_loss:.6f}")
                print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                if train_cont is not None:
                    print(f"Train Terms - {self._format_dae_terms(train_cont)}")
                for loader_name, (_, val_terms) in val_results.items():
                    print(f"Val[{loader_name}] Terms - {self._format_dae_terms(val_terms)}")
                print(
                    f"Epoch Progress: {epoch+1}/{training_config.get('epochs', 200)} "
                    f"({(epoch+1)/training_config.get('epochs', 200)*100:.1f}%)"
                )
                print("-" * 50)
            is_best = monitor_val_loss < self.best_val_loss if monitor_val_loss is not None else False
            if is_best:
                self.best_val_loss = monitor_val_loss
                if is_main_process(self.rank):
                    print(
                        "New best validation loss: "
                        f"{monitor_val_loss:.6f} ({self.monitor_val_name})"
                    )
            checkpoint_interval = training_config.get('checkpoint_interval', 5)
            if (epoch + 1) % checkpoint_interval == 0 or is_best:
                if is_main_process(self.rank):
                    self.checkpoint_manager.save_checkpoint(
                        epoch + 1, self.optimizer, self.scheduler,
                        self.train_losses, self.val_losses, self.best_val_loss,
                        self.config, is_best, best_tag=self.monitor_val_name
                    )
        if is_main_process(self.rank):
            self.checkpoint_manager.save_checkpoint(
                training_config.get('epochs', 200), self.optimizer, self.scheduler,
                self.train_losses, self.val_losses, self.best_val_loss,
                self.config, is_best=False, best_tag=self.monitor_val_name
            )
            print("Training completed!")

    def load_checkpoint(self, checkpoint_path):
        train_losses, val_losses, best_val_loss, start_epoch = (
            self.checkpoint_manager.load_checkpoint(
                checkpoint_path, self.optimizer, self.scheduler, self.device
            )
        )

        self.train_losses = train_losses
        self.val_losses = val_losses
        self.best_val_loss = best_val_loss

        return start_epoch

    def detect_weight_anomalies(self, step):
        WeightAnomalyDetector.detect_anomalies(self.model, step)
