
# coding: utf-8

# In[1]:

import torch
from torch.autograd import Variable


# In[2]:

_BIG_NEGATIVE = -1000000.0


# In[3]:

def lstm_cell_hidden(mprev, cprev, node_dim, attention_m=False):
    """
    Create an LSTM cell.

    The way this LSTM cell is
    used, there is no input x, instead the m and c are updated according to the
    LSTM equations treating the input x as the zero vector. However the m at each
    time step is concatenated with an external input as described in
    https://arxiv.org/pdf/1511.06391.pdf.

    Implements the equations in pg.2 from
    "Long Short-Term Memory Based Recurrent Neural Network Architectures
    For Large Vocabulary Speech Recognition",
    Hasim Sak, Andrew Senior, Francoise Beaufays.

    Args:
    mprev: m_{t-1}, the recurrent activations (same as the output)
      from the previous cell.
    cprev: c_{t-1}, the cell activations from the previous cell.
    node_dim: Number of hidden state of the LSTM.
    attention_m: If true then the hidden dim is twice the size of the cell dim
    name: prefix for the variable names

    Returns:
    m: Outputs of this cell.
    c: Cell Activations.
    """

    # Input Gate
    m_nodes = node_dim
    if attention_m:
        m_nodes = 2 * node_dim
    im = Variable(torch.rand(m_nodes,node_dim))
    ib = Variable(torch.zeros(1,node_dim))
    i_g  = torch.sigmoid(torch.matmul(mprev,im) + ib)
    
    #Forget Gate
    fm = Variable(torch.rand(m_nodes,node_dim))
    fb = Variable(torch.zeros(1,node_dim))
    f_g = torch.sigmoid(torch.matmul(mprev,fm) + fb)
    
    #Cell
    cm = Variable(torch.rand(m_nodes,node_dim))
    cb = Variable(torch.zeros(1,node_dim))
    cprime = torch.sigmoid(torch.matmul(mprev,cm) + cb)
    c = f_g * cprev + i_g * torch.tanh(cprime)
    
    #Output Gate
    om = Variable(torch.rand(m_nodes,node_dim))
    ob = Variable(torch.zeros(1,node_dim))
    o_g = torch.sigmoid(torch.matmul(mprev,om) + ob)
    m = o_g * torch.tanh(c)
    return m,c


# In[4]:

def set2vec(input_set,
            num_timesteps,
            mprev=None,
            cprev=None,
            mask=None,
            inner_prod="default"):
    """
  Part of the set2set model described in Vinyals et. al.

  Specifically this implements the "process" block described in
  https://arxiv.org/pdf/1511.06391.pdf. This maps a set to a single embedding
  m which is invariant to the order of the elements in that set. Thus it should
  be thought of as a "set2vec" model. It is part of the full set2set model from
  the paper.

  There is an LSTM which from t = 1,...,num_timesteps emits a query vector at
  each time step, which is used to perform content based attention over the
  embedded input set (see https://arxiv.org/pdf/1506.03134.pdf sec 2.2), and
  the result of that content based attention is then fed back into the LSTM
  by concatenation with m, the output of the LSTM at that time step. After
  num_timesteps of computation we return the final cell c, and output m.
  m can be considered the order invariant embedding of the input_set.

  Args:
    input_set: tensor of shape [batch_size, num_nodes, 1, node_dim]
    num_timesteps: number of computation steps to run the LSTM for
    mprev: Used to initialize the hidden state of the LSTM, pass None if
      the hidden state should be initialized to zero.
    cprev: Used to initialize the cell of the LSTM, pass None if the cell
      state should be initialized to zero.
    mask: tensor of type bool, shape = [batch_size,num_nodes]. This is
      used when batches may contain sets of different sizes. The values should
      be binary. If set to None then the model will assume all sets have the
      same size.
    inner_prod: either 'default' or 'dot'. Default uses the attention mechanism
      as described in the pointer networks paper. Dot is standard dot product.
      The experiments for the MPNN paper (https://arxiv.org/pdf/1704.01212.pdf)
      did not show a significant difference between the two inner_product types,
      and the final experiments were run with default.
    name: (string)

  Returns:
    logit_att: A list of the attention masks over the set.
    c: The final cell state of the internal LSTM.
    m: The final output of the internal LSTM (note this is what we use as the
      order invariant representation of the set).

  Raises:
    ValueError: If an invalid inner product type is given.
  """
    batch_size = input_set.shape[0]
    node_dim = input_set.shape[3]
    # For our use case the "input" to the LSTM at each time step is the
    # zero vector, instead the hidden state of the LSTM at each time step
    # will be concatenated with the output of the content based attention
    # (see eq's 3-7 in the paper).
    if mprev is None:
        mprev = torch.zeros(batch_size,node_dim)
    mprev = torch.cat((mprev,torch.zeros(batch_size,node_dim)),1)
    
    if cprev is None:
        cprev = torch.zeros(batch_size,node_dim)
    
    
    logit_att = []
    attention_w2 = Variable(torch.rand(node_dim,node_dim))
    attention_v = Variable(torch.rand(node_dim,1))
    # Batches may contain sets of different sizes, in which case the smaller
    # sets will be padded with null elements as specified by the mask.
    # In order to make the set2vec model invariant to this padding, we add
    # large negative numbers to the logits of the attention softmax (which when
    # exponentiated will become 0).
    if mask is not None:
        mask = (1 - mask) * _BIG_NEGATIVE
        
    for i in range(num_timesteps):
        m,c = lstm_cell_hidden(
               mprev,cprev,node_dim,attention_m=True)
        query = torch.matmul(m,attention_w2)
        query = torch.reshape(query,(-1,1,1,node_dim))
        if inner_prod == 'default':
            energies = torch.reshape(
                        torch.matmul(
                                torch.reshape(torch.tanh(query+input_set),(-1,node_dim)),
                                attention_v),(batch_size,-1))
        elif inner_prod == 'dot':
            att_mem_reshape = torch.reshape(input_set,(batch_size,-1,node_dim))
            query = torch.reshape(query,(-1,node_dim,1))
            energies = torch.reshape(
                        torch.matmul(att_mem_reshape,query),(batch_size,-1))
        else:
            raise ValueError("Invalid inner_prod type: {}".format(inner_prod))
        #Zero out non nodes
        if mask is not None:
            energies += mask
        att = torch.nn.softmax(energies)
        
        #multiply attention mask over the elements of the set
        att = torch.reshape(att,(batch_size,-1,1,1))
        # Take the weighted average the elements in the set
        # This is the 'r' of the paper.
        read_mult = att * input_set
        read = read_mult.sum(1).sum(2)
        m = torch.cat((m,read),1)
        
        logit_att.append(m)
        mprev = m
        cprev = c
        return logit_att,c,m


# In[ ]:



