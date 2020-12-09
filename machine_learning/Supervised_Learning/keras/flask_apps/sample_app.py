#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask


# In[2]:


app = Flask(__name__)


# In[4]:


@app.route('/sample')
def running():
    return 'Flask is running!'

