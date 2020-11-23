import os.path as pth
import json
from visdom import Visdom
import numpy as np

class VisdomLinePlot():

	def __init__(self, env_name='main', server="0.0.0.0", port=8899):
		self.viz = Visdom(
			port=port,
			env=env_name,
            server=server
		)
		self.plot_list = {}
		self.env = env_name
	def plotLine(self, scalar_name, split, title_name, x ,y):

		if scalar_name not in self.plot_list:

			self.plot_list[scalar_name] = self.viz.line( X=np.array([x,x]), Y=np.array([y,y]), env=self.env,
														 opts=dict(legend=[split],
																   title=title_name,
														  			xlabel='Epochs',
																	ylabel= scalar_name))
		else:

			self.viz.line(X=np.array([x]), Y=np.array([y]),
						  env=self.env,
						  win=self.plot_list[scalar_name],
						  name=split, update='append')

	def writeText(self, dict):
		output = ''
		for arg in vars(dict):
			output=output+('{:<20}: {}{}'.format(arg, getattr(dict, arg),"\n"))
		self.viz.text(output)