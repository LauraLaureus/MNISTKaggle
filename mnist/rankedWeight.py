import numpy
import OneHotConverter

def parseToOneHot(array):
    i = numpy.argmax(array)
    result = numpy.zeros(array.shape)
    result[i] = 1
    return result

#testVector = [1,2,3,1,2,3]
#print(softmaxWeights(testVector))

gradient = 0.96914
closed = 0.97029
canny = 0.97914
skel = 0.74957
extensive = 0.99214
#normal = 0.99114
accuracy = [ extensive,skel,canny,closed,gradient]

weights = [1,1/5,1/2,1/3,1/4]
print(weights)

fileNames = ['./modelos/03-extensivetrain/cnn_extensive.txt','./modelos/03b-skel entrenado/cnn_skel.txt','./modelos/04 - canny 10000/cnn_canny.txt','./modelos/05 - closed 10000/cnn_closed.txt', './modelos/06 - gradient 10000/cnn_gradient.txt']
result = numpy.zeros((28000,10))

for i in range(len(fileNames)):
    db = numpy.loadtxt(fileNames[i],delimiter=',',ndmin=2)
    for r in range(db.shape[0]):
        result[r,:] = result[r,:] + db[r,:] * weights[i]
        

print(result[0,:])

for r in range(result.shape[0]):
    result[r,:] = parseToOneHot(result[r,:])

outputFileName = 'rankedAssemblyDecisionOH.txt'
numpy.savetxt(outputFileName,result,delimiter=',')
OneHotConverter.convert(outputFileName)
