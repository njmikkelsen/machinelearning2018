import sys
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score

class computations(object):
  """
  A module with wrappers for various functions and computation.
  The following is short list of the module's content:
    metrics: MSE,R2,accuracy
    mat_ops: SVD,
  """
  class metrics(object):
    """
    Some common metrics used in machine learning,
    both regression and classification.
    """
    # Mean Squared Error
    def MSE(T,Y):
      return mean_squared_error(T,Y)
    # Coefficient of Determination
    def R2(T,Y):
      return r2_score(T,Y)
    # label accuracy
    def accuracy(T,Y):
      return accuracy_score
  
  class array_ops(object):
    """
    Some useful shorthands for matrix & vector operations.
    """
    # Singular Value Decomposition
    def SVD(matrix):
      return svd(matrix,full_matrices=False,check_finite=False,lapack_driver="gesdd")

    # one-hot in numpy
    def to_categorical_numpy(integer_vector):
      """
      This function was copied from the lecture notes on Neural Networks found here:
      https://compphysics.github.io/MachineLearning/doc/pub/NeuralNet/html/._NeuralNet-bs046.html
      """
      n_inputs = len(integer_vector)
      n_categories = np.max(integer_vector) + 1
      onehot_vector = np.zeros((n_inputs, n_categories))
      onehot_vector[range(n_inputs), integer_vector] = 1
      return onehot_vector
    
    

class progress_bar(object):
  """
  A command-line progress bar for use in large loops (as a sanity check).
  Argument N = number of loop iterations. Usage:
  -------------------------------
  BAR = misclib.progress_bar(N)
  for i in range(N):
    # do something
    BAR.update()
  -------------------------------
  """
  def __init__(self,N):
    self.n = 0
    self.N = N
    print('')
    sys.stdout.write('\r['+' '*20+']    0 %')
  def update(self)
    self.n += 1
    if self.n < N:
      sys.stdout.write('\r[{:20s}] {:4.0f} % '.format('='*int(20*self.n/self.N),100.*self.n/self.N))
    else:
      sys.stdout.write('\r['+'='*20+']  100 % Done!\n')








