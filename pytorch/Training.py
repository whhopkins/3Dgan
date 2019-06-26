from HDF5torch import HDF5generator
import numpy as np
import glob
import time

class TwoLayerNet(torch.nn.Module):
   def __init__(self, D_in, H, D_out):
      """
      In the constructor we instantiate two nn.Linear modules and assign them as
      member variables.
      """
      super(TwoLayerNet, self).__init__()
      self.linear1 = torch.nn.Linear(D_in, H)
      self.linear2 = torch.nn.Linear(H, D_out)

   def forward(self, x):
      """
      In the forward function we accept a Tensor of input data and we must return
      a Tensor of output data. We can use Modules defined in the constructor as
      well as arbitrary operators on Tensors.
      """
      h_relu = self.linear1(x).clamp(min=0)
      y_pred = self.linear2(h_relu)

      return y_pred

def training():
   base_path = "/bigdata/shared/LCD/NewV1/" # fixed angle                                                                                         
   particle =['Ele', 'Pi0'] # particle types                                                                                                      
   sample_path = []
   for p in particle:
     sample_path.append(base_path + p + 'Escan/*.h5')

   batch_size = 128
   train_ratio = 0.9
   shuffle=True

   n_classes = len(particle)
   # gather sample files for each type of particle                                                                                                
   class_files = [[]] * n_classes
   for i, class_path in enumerate(sample_path):
      class_files[i] = sorted(glob.glob(class_path))
   files_per_class = min([len(files) for files in class_files])

   # split the train, test, and validation files                                                                                                  
   # get lists of [[class1_file1, class2_file1], [class1_file2, class2_file2], ...]                                                               
   train_files = []
   test_files = []
   n_train = int(files_per_class * train_ratio)
   for i in range(files_per_class):
       new_files = []
       for j in range(n_classes):
           new_files.append(class_files[j][i])
           if i < n_train:
               train_files.append(new_files)
           else:
               test_files.append(new_files)
   init = time.time()
   # data generator for train files                                                                                                               
   data_gen = HDF5generator(train_files[:3], batch_size=batch_size, shuffle=shuffle) #, num_events=5000)                                          
   data_gen.particles=particle
   print('Initialization in {} sec'.format(time.time()-init))
   init = time.time()
   for key in data_gen.data:
       print(key, data_gen.data[key].shape)
   for i, data in enumerate(data_gen):
      print(i)
      for key in data:
        if key =='ECAL':
            print(key, data[key].shape)
        else:
            print(key, data[key][:5])
      print('{} batches took {} sec'.format(i, time.time()-init))
   print('{} batches loaded in {} sec'.format(i, time.time()-init))


if __name__ == "__main__":
    training()
