# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import random
from tensorflow.keras.models import load_model,Model 


class SFO(tf.keras.optimizers.Optimizer):

    def __init__(self,learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-7,
                  amsgrad=False,weight_decay=None,clipnorm=None,clipvalue=None,
                  global_clipnorm=None,use_ema=False,ema_momentum=0.99,
                  ema_overwrite_frequency=None,jit_compile=True,name="SFO",**kwargs):
        super().__init__(name=name,weight_decay=weight_decay,clipnorm=clipnorm,
                          clipvalue=clipvalue,global_clipnorm=global_clipnorm,use_ema=use_ema,
                          ema_momentum=ema_momentum,ema_overwrite_frequency=ema_overwrite_frequency,
                          jit_compile=jit_compile, **kwargs )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        # self.iterations=100
        
    def objective_function(x):
        selected_labels = audio_y_tr[:, :5] 
        cnf_matrix = confusion_matrix(audio_y_tr, audio_model)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)    
        ACC = (TP+TN)/(TP+FP+FN+TN)
        Accuracy=sum(ACC)/len(ACC)    
        return Accuracy

        
    
    def optimize(self,Dim,variable_bounds,ub,lb,population_size=30, generations=100, alpha=0.5,):
        # population = np.random.uniform(-1, 1, size=(population_size, num_dimensions))
        num_variables = len(variable_bounds)
        lb=variable_bounds[0, 0]
        ub=variable_bounds[0, 1]
        # Initialize the population
        set_global=0
        Sheep = np.random.uniform(lb,ub, size=(population_size, num_variables))
        iteration=generations*0.1
        XGBest=Sheep[0]
        XLbest=Sheep[-1]
        while iteration<=generations:
            T=1-(iteration/generations)
            
            
            for i in population_size:
                
                  if iteration >= generations - 5:
                    
                      rGsheep=2*0.001*(ub-lb)*T  #eqn--------4
                      rGgoat =0.1*(ub-lb)*T    #eqn--------5
                     
                      Rand=random.uniform(0.1)
                      dx=(2*rGsheep)*Rand- (rGgoat)       #eqn-------3
                      x=Sheep[i]+dx
                     
                      cost=self.objective_function(Sheep[i])
                
                      if cost > set_global:
                        set_global=cost
                        Sheep[i]=x
                        XGBest=Sheep[i]
                        
                  else:
                    XRandomSheep=Sheep[random.choice(0,population_size)]
                    X=Sheep[i]
                    Rand=random.uniform(0,1)
                    C = 3*random.uniform(0,1)     
                    vsh1 = (1-T)*C*Rand(1-Dim)*(XGBest - X) #eqn-------(7)
                    vLbest1 = C*Rand(1, Dim)*(XLbest - X)     #eqn-------8
                    vother = C*Rand(1 , Dim)*(XRandomSheep - X)        #eqn-------9
                    C1 = 3*random.uniform(0,1)  
                    vsh21 = C1* (1 - T)*(XGBest - X)   #eqn-------(11)       
                    vsh12= Rand(1, Dim)*(XGBest - X)   #eqn--------(14)                      
                    vLbest2= (1 - T)*2*Rand(1, Dim)* (XLbest - X ) #eqn----(15)    
                    vsh22= (1 - T)*2*Rand(1, Dim)*XGBest - X   #eqn----------(16) 
                    
                    
                    if T > 0.3:
                        Vm1 = vsh1 + vLbest1 + vother
                     
                    elif T <= 0.3:
                        Vm1 = vsh21 + vLbest1
                        
                    
                    if T > 0.7:
                        Vm2 = vsh12 + vLbest2
                        
                    elif T <= 0.7:   
                        Vm2 = vsh22
                        
    
    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    )
                )
    
    def update_step(self, gradient, variable):
        population_size = 50
        num_dimensions = 10
        num_iterations = 100
        pbmsize=2;
        lb=-5;
        ub=5;
        variable_bounds = np.array([[lb, ub]] * pbmsize)  # Example bounds for 10 variables     
        # self.optimize(pbmsize,variable_bounds,ub,lb)
        """Update step given gradient and the associated model variable."""
        beta_1_power = None
        beta_2_power = None
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)

        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        
        v = self._velocities[self._index_dict[var_key]] 
        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)
     
        if isinstance(gradient, tf.IndexedSlices):
           
            m.assign_add(-m * (1 - self.beta_1))
           
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.beta_1), gradient.indices
                )
            )
            
            v.assign_add(-v * (1 - self.beta_2))
          
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2),
                    gradient.indices,
                )
            )
            
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        else:
            
            m.assign_add((gradient - m) * (1 - self.beta_1))
           
            v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))
            
            if self.amsgrad:
               
                v_hat = self._velocity_hats[self._index_dict[var_key]]
               
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
           
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
            
    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config
    



