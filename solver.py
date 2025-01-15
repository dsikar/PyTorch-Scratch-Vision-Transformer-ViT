import os
import sys
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from data_loader import get_loader
from sklearn.metrics import confusion_matrix, accuracy_score
from model import VisionTransformer, VisionTransformer_pytorch

class Solver(object):
    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.best_acc = 0

        # Get data loaders
        self.train_loader, self.test_loader = get_loader(args)

        # Initialize model
        self.model = self._create_model()
        
        # Save training configuration and parameters - moved after model creation
        self.save_training_notes()
        
        # Initialize tracking arrays
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        
        # Setup optimizer and schedulers
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-3)
        self.schedulers = self._create_schedulers()
        
        # Resume from checkpoint if specified
        if args.resume:
            self._load_checkpoint(args.resume)

    def _create_model(self):
        if self.args.use_torch_transformer_layers:
            model = VisionTransformer_pytorch(n_channels=self.args.n_channels,   embed_dim=self.args.embed_dim, 
                                            n_layers=self.args.n_layers,       n_attention_heads=self.args.n_attention_heads, 
                                            forward_mul=self.args.forward_mul, image_size=self.args.image_size, 
                                            patch_size=self.args.patch_size,   n_classes=self.args.n_classes, 
                                            dropout=self.args.dropout)
        else:
            model = VisionTransformer(n_channels=self.args.n_channels,   embed_dim=self.args.embed_dim, 
                                    n_layers=self.args.n_layers,       n_attention_heads=self.args.n_attention_heads, 
                                    forward_mul=self.args.forward_mul, image_size=self.args.image_size, 
                                    patch_size=self.args.patch_size,   n_classes=self.args.n_classes, 
                                    dropout=self.args.dropout)

        # Push to GPU
        if self.args.is_cuda:
            model = model.cuda()

        # Display Vision Transformer
        print('--------Network--------')
        print(model)       

        # Training parameters stats
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in the model: {n_parameters}")

        # Option to load pretrained model
        if self.args.load_model:
            print("Using pretrained model")
            model.load_state_dict(torch.load(os.path.join(self.args.model_path, 'ViT_model.pt')))

        # Training loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        return model

    def _create_schedulers(self):
        linear_warmup = optim.lr_scheduler.LinearLR(
            self.optimizer, 
            start_factor=1/self.args.warmup_epochs,
            end_factor=1.0,
            total_iters=self.args.warmup_epochs-1,
            last_epoch=-1,
            verbose=True
        )
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.args.epochs-self.args.warmup_epochs,
            eta_min=1e-5,
            verbose=True
        )
        return [linear_warmup, cos_decay]

    def _load_checkpoint(self, checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Verify compatibility
        for key, value in vars(checkpoint['args']).items():
            if key in ['resume', 'checkpoint_frequency']:  # Skip certain args
                continue
            if getattr(self.args, key) != value:
                print(f"Warning: Argument mismatch for {key}: checkpoint={value}, current={getattr(self.args, key)}")
        
        # Load model and training state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler states if they exist
        if checkpoint['linear_warmup_state']:
            self.schedulers[0].load_state_dict(checkpoint['linear_warmup_state'])
        if checkpoint['cos_decay_state']:
            self.schedulers[1].load_state_dict(checkpoint['cos_decay_state'])
        
        # Restore training progress
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.test_accuracies = checkpoint['test_accuracies']
        
        print(f"Resuming from epoch {self.start_epoch}")

    def save_checkpoint(self, epoch, optimizer, schedulers, best_acc, checkpoint_path):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'linear_warmup_state': schedulers[0].state_dict() if epoch < self.args.warmup_epochs else None,
            'cos_decay_state': schedulers[1].state_dict() if epoch >= self.args.warmup_epochs else None,
            'best_acc': best_acc,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'args': self.args
        }
        torch.save(checkpoint, checkpoint_path)

    def test_dataset(self, loader):
        # Set Vision Transformer to evaluation mode
        self.model.eval()

        # Arrays to record all labels and logits
        all_labels = []
        all_logits = []

        # Testing loop
        for (x, y) in loader:
            if self.args.is_cuda:
                x = x.cuda()

            # Avoid capturing gradients in evaluation time for faster speed
            with torch.no_grad():
                logits = self.model(x)

            all_labels.append(y)
            all_logits.append(logits.cpu())

        # Convert all captured variables to torch
        all_labels = torch.cat(all_labels)
        all_logits = torch.cat(all_logits)
        all_pred   = all_logits.max(1)[1]
        
        # Compute loss, accuracy and confusion matrix
        loss = self.loss_fn(all_logits, all_labels).item()
        acc  = accuracy_score(y_true=all_labels, y_pred=all_pred)
        cm   = confusion_matrix(y_true=all_labels, y_pred=all_pred, labels=range(self.args.n_classes))

        return acc, cm, loss

    def test(self, train=True):
        if train:
            # Test using train loader
            acc, cm, loss = self.test_dataset(self.train_loader)
            print(f"Train acc: {acc:.2%}\tTrain loss: {loss:.4f}\nTrain Confusion Matrix:")
            print(cm)

        # Test using test loader
        acc, cm, loss = self.test_dataset(self.test_loader)
        print(f"Test acc: {acc:.2%}\tTest loss: {loss:.4f}\nTest Confusion Matrix:")
        print(cm)

        return acc, loss

    def train(self):
        iters_per_epoch = len(self.train_loader)
        
        # Track last checkpoint for cleanup
        last_checkpoint_path = None

        # Training loop
        for epoch in range(self.start_epoch, self.args.epochs):

            # Set model to training mode
            self.model.train()

            # Arrays to record epoch loss and accuracy
            train_epoch_loss     = []
            train_epoch_accuracy = []

            # Loop on loader
            for i, (x, y) in enumerate(self.train_loader):

                # Push to GPU
                if self.args.is_cuda:
                    x, y = x.cuda(), y.cuda()

                # Get output logits from the model 
                logits = self.model(x)

                # Compute training loss
                loss = self.loss_fn(logits, y)

                # Updating the model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Batch metrics
                batch_pred            = logits.max(1)[1]
                batch_accuracy        = (y==batch_pred).float().mean()
                train_epoch_loss     += [loss.item()]
                train_epoch_accuracy += [batch_accuracy.item()]

                # Log training progress
                if i % 50 == 0 or i == (iters_per_epoch - 1):
                    print(f'Ep: {epoch+1}/{self.args.epochs}\tIt: {i+1}/{iters_per_epoch}\tbatch_loss: {loss:.4f}\tbatch_accuracy: {batch_accuracy:.2%}')

            # Test the test set after every epoch
            # Checkpointing logic
            if self.args.checkpoint_frequency > 0 and (epoch + 1) % self.args.checkpoint_frequency == 0:
                # Delete previous checkpoint if it exists
                if last_checkpoint_path and os.path.exists(last_checkpoint_path):
                    os.remove(last_checkpoint_path)
                
                # Create new checkpoint
                checkpoint_name = f"{self.args.timestamp_prefix}checkpoint_epoch_{epoch+1}.pt"
                checkpoint_path = os.path.join(self.args.model_path, checkpoint_name)
                self.save_checkpoint(
                    epoch,
                    self.optimizer,
                    self.schedulers,
                    self.best_acc,
                    checkpoint_path
                )
                last_checkpoint_path = checkpoint_path
                print(f"Saved checkpoint at epoch {epoch+1}: {checkpoint_path}")

            test_acc, test_loss = self.test(train=((epoch+1)%25==0)) # Test training set every 25 epochs

            # Capture best test accuracy
            self.best_acc = max(test_acc, self.best_acc)
            print(f"Best test acc: {self.best_acc:.2%}\n")

            # Only save final model on last epoch
            if epoch == self.args.epochs - 1:
                model_name = f"{self.args.timestamp_prefix}final_model_acc_{self.best_acc:.2f}.pt"
                torch.save(self.model.state_dict(), os.path.join(self.args.model_path, model_name))
                
                # Delete the last checkpoint if it exists since we have the final model
                if last_checkpoint_path and os.path.exists(last_checkpoint_path):
                    os.remove(last_checkpoint_path)
            
            # Update learning rate using schedulers
            if epoch < self.args.warmup_epochs:
                self.schedulers[0].step()
            else:
                self.schedulers[1].step()

            # Update training progression metric arrays
            self.train_losses     += [sum(train_epoch_loss)/iters_per_epoch]
            self.test_losses      += [test_loss]
            self.train_accuracies += [sum(train_epoch_accuracy)/iters_per_epoch]
            self.test_accuracies  += [test_acc]


    def plot_graphs(self):
        # Plot graph of loss values
        plt.plot(self.train_losses, color='b', label='Train')
        plt.plot(self.test_losses, color='r', label='Test')

        plt.ylabel('Loss', fontsize = 18)
        plt.yticks(fontsize=16)
        plt.xlabel('Epoch', fontsize = 18)
        plt.xticks(fontsize=16)
        plt.legend(fontsize=15, frameon=False)

        # plt.show()  # Uncomment to display graph
        plt.savefig(os.path.join(self.args.output_path, 'graph_loss.png'), bbox_inches='tight')
        plt.close('all')


        # Plot graph of accuracies
        plt.plot(self.train_accuracies, color='b', label='Train')
        plt.plot(self.test_accuracies, color='r', label='Test')

        plt.ylabel('Accuracy', fontsize = 18)
        plt.yticks(fontsize=16)
        plt.xlabel('Epoch', fontsize = 18)
        plt.xticks(fontsize=16)
        plt.legend(fontsize=15, frameon=False)

        # plt.show()  # Uncomment to display graph
        plt.savefig(os.path.join(self.args.output_path, 'graph_accuracy.png'), bbox_inches='tight')
        plt.close('all')

    def save_training_notes(self):
        notes_filename = f"{self.args.timestamp_prefix}notes.txt"
        notes_path = os.path.join(self.args.model_path, notes_filename)
        
        # Prepare notes content
        notes = [
            "=" * 80,  # Separator
            f"Training started: {self.args.timestamp_prefix[:-1]}",  # Remove trailing underscore for display
            "\nCommand Line Arguments:",
            " ".join(sys.argv),  # Capture command line command
            "\nModel Parameters:",
            f"Total trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}",
            "\nHyperparameters:",
        ]
        
        # Add all args
        for k, v in vars(self.args).items():
            notes.append(f"{k}: {v}")
        
        # Write mode: 'a' for append if resuming, 'w' for new file
        mode = 'a' if self.args.resume else 'w'
        with open(notes_path, mode) as f:
            f.write('\n'.join(notes) + '\n\n')

