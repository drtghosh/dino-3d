import os
import torch
import torch.optim as optim
import torch.nn as nn
from abc import abstractmethod
from tensorboardX import SummaryWriter

from models import get_model

class TrainWatcher(object):
	"""
		Class of object to track epoch and step during training
	"""
	def __init__(self):
		self.epoch = 0
		self.minibatch = 0
		self.step = 0

	def within_epoch(self):
		self.minibatch += 1
		self.step += 1

	def new_epoch(self):
		self.epoch += 1
		self.minibatch = 0

	def make_checkpoint(self):
		return {
			'epoch': self.epoch,
			'minibatch': self.minibatch,
			'step': self.step
		}

	def restore_checkpoint(self, clock_dict):
		self.epoch = clock_dict['epoch']
		self.minibatch = clock_dict['minibatch']
		self.step = clock_dict['step']
    

class TrainerDINO(object):
	"""
		DINO trainer that provides training process with student and techer models.
	"""
	def __init__(self, config):
		# paths/ directories
		self.log_dir = config.log_dir
		self.model_dir = config.model_dir

		# device
		self.device = config.device

		# watcher
		self.watcher = TrainWatcher()

		# training batch size
		self.batch_size = config.batch_size

		# build network
		self.teacher_model, self.student_model = self.build_model(config)

		# set loss function
		self.criterion = None
		self.set_loss_function()

		# set optimizer
		self.base_lr = None
		self.optimizer = None
		self.set_optimizer(config)

		# set lr scheduler
		self.scheduler = None
		self.set_scheduler(config)

		# store predicted probabilities
		self.student_probs = None
		self.teacher_probs = None

		# store loss values
		self.loss = None

		# set tensorboard writer
		self.train_tbw = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
		self.val_tbw = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

	@abstractmethod
	def build_model(self, config):
		teacher = get_model(config, "dinov2")
		student = get_model(config, "dinov2")
		teacher = teacher.to(self.device)
		student = student.to(self.device)
		return teacher, student

	def set_loss_function(self):
		"""
			set loss function used in training
		"""
		self.criterion = nn.CrossEntropyLoss().to(self.device)
		
	@abstractmethod
	def forward(self, data):
		"""
			forward method for the student and teacher networks
		"""
		input_global = data["global_view"].to(self.device)
		input_local = data["local_views"].to(self.device)
		

		

	@abstractmethod
	def collect_loss(self):
		"""
			collect all losses into a dict
		"""
		raise NotImplementedError

	def set_optimizer(self, config):
		"""
			set optimizer used in training
		"""
		self.base_lr = config.lr
		self.optimizer = optim.Adam(self.model.parameters(), config.lr)

	def set_scheduler(self, config):
		"""
			set lr scheduler used in training
		"""
		# self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, config.lr_step_size)
		self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.lr_decay)

	def save_ckpt(self, name=None):
		"""
			save checkpoint during training for future reload
		"""
		if name is None:
			save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.watcher.epoch))
			print("Saving checkpoint epoch {}...".format(self.watcher.epoch))
		else:
			save_path = os.path.join(self.model_dir, "{}.pth".format(name))

		if isinstance(self.model, nn.DataParallel):
			model_state_dict = self.model.module.cpu().state_dict()
		else:
			model_state_dict = self.model.cpu().state_dict()

		torch.save({
			'watcher': self.watcher.make_checkpoint(),
			'model_state_dict': model_state_dict,
			'optimizer_state_dict': self.optimizer.state_dict(),
			'scheduler_state_dict': self.scheduler.state_dict(),
		}, save_path)

		self.model.to(self.device)

	def load_ckpt(self, epoch, name=None):
		"""
			load checkpoint from saved checkpoint
		"""
		name = name if name == 'latest' else "ckpt_epoch{}".format(epoch)
		load_path = os.path.join(self.model_dir, "{}.pth".format(name))
		if not os.path.exists(load_path):
			raise ValueError("Checkpoint {} not exists.".format(load_path))

		checkpoint = torch.load(load_path)
		print("Loading checkpoint from {} ...".format(load_path))
		if isinstance(self.model, nn.DataParallel):
			self.model.module.load_state_dict(checkpoint['model_state_dict'])
		else:
			self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		self.watcher.restore_checkpoint(checkpoint['watcher'])

	def update_network(self, loss_dict):
		"""
			update network by back propagation
		"""
		loss = sum(loss_dict.values())
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def update_learning_rate(self):
		"""
			record and update learning rate
		"""
		self.train_tbw.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], self.watcher.epoch)
		if not self.optimizer.param_groups[-1]['lr'] < self.base_lr / 10.0:
			self.scheduler.step(self.watcher.epoch)

	def record_losses(self, loss_dict, mode='train'):
		"""
			record loss to tensorboard
		"""
		losses_values = {k: v.item() for k, v in loss_dict.items()}

		tbw = self.train_tbw if mode == 'train' else self.val_tbw
		for k, v in losses_values.items():
			tbw.add_scalar(k, v, self.watcher.step)

	def train_func(self, data):
		"""
			one step of training
		"""
		self.model.train()
		self.forward(data)

		losses = self.collect_loss()
		self.update_network(losses)
		self.record_losses(losses, 'train')

	def val_func(self, data):
		"""
			one step of validation
		"""
		self.model.eval()

		with torch.no_grad():
			self.forward(data)

		losses = self.collect_loss()
		self.record_losses(losses, 'validation')

	def visualize_batch(self, data, tbw, num, **kwargs):
		"""
			write visualization results to tensorboard writer
		"""
		raise NotImplementedError