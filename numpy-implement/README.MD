# handwritten digtis classification

## How to run?
**Train** run `python train.py` to train network, you can tune hyperpameters in utils top part  
**Test** run `python test_set.py` to load trained weights and see the accuracy in train set and test set  
**predict** put handwritten digits in test-images folder and run `python predict_image.py` to see predict value

## Structure and results
*easy 2 layers neural netwroks*  
*Linear -> Relu -> Batch Norm -> Liner -> Softmax*  
after running 2000 epochs, the result is acceptable:  
__*train accuracy*__ = 0.9997  
__*test accuracy*__ = 0.9707

  
