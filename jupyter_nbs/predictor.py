
# coding: utf-8

# In[1]:


import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import time
import argparse
import os
import sys
import numpy as np

# Our Modules
reader_folder = os.path.realpath(os.path.abspath('..'))
if reader_folder not in sys.path:
    sys.path.append(reader_folder)
import datasets
from datasets import utils
from models.MPNN import MPNN
from LogMetric import AverageMeter, Logger


# In[2]:


root = os.path.join('../../../QMfiles/ts_xyz/')
files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]


# In[37]:


files


# In[3]:


idx = np.random.permutation(len(files))
idx = idx.tolist()


# In[4]:


len(idx)


# In[5]:


test_ids = [files[i] for i in idx[:100]]


# In[6]:


len(test_ids)


# In[7]:


data_test = datasets.ts(root, test_ids)


# In[8]:


# Define model and optimizer
print('Define model')
# Select one graph
g_tuple, l = data_test[0]
g, h_t, e = g_tuple

print('\tStatistics')
stat_dict = datasets.utils.get_graph_stats(data_test, ['degrees', 'target_mean', 'target_std', 'edge_labels'])


# In[9]:


data_test.set_target_transform(lambda x: datasets.utils.normalize_data(x, stat_dict['target_mean'],
                                                                           stat_dict['target_std']))


# In[10]:


test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=20, collate_fn=datasets.utils.collate_g,
                                              num_workers=4, pin_memory=True)


# In[11]:


model = MPNN([len(h_t[0]), len(list(e.values())[0])], 73, 15, 2, len(l), type='regression')


# In[12]:


model_pth = torch.load('../checkpoint/ts/mpnn/model_best.pth')


# In[13]:


model.load_state_dict(model_pth['state_dict'])


# In[14]:


model.parameters


# In[15]:


model = model.cuda()


# In[16]:


model = model.train()


# In[17]:


output_list =[]
for i , (g,h,e,target) in enumerate(test_loader):
    g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
    g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)
    output = model(g, h, e)
    output_list.append(output)


# In[22]:


print output_list


# In[27]:


batch_1 = output_list


# In[29]:


batch_1 = np.array(batch_1.tolist())


# In[33]:


(batch_1* stat_dict['target_std']) + stat_dict['target_mean']


# In[34]:


import pandas as pd


# In[35]:


param = pd.read_csv('../../../QMfiles/two_param.csv')


# In[41]:


for i in idx[:100]:
    print param.ea2[i]


# In[47]:


array_list = []
for batch in output_list:
    array_list.append(batch.tolist())


# In[48]:


array_list = np.array(array_list)
array_list = (array_list* stat_dict['target_std']) + stat_dict['target_mean']


# In[52]:


array_list

