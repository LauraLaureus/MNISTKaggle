import numpy

def convert(file = 'cnn_output.txt'):
    oneHotRegistry = numpy.loadtxt(file,delimiter=",",ndmin=2)

    def translateOneHot(oh):
        for i in range(10):
            if oh[i] == 1:
                return i
        return 0

    #Allocate space for array
    data = numpy.fromiter(range(oneHotRegistry.shape[0]), count=oneHotRegistry.shape[0], dtype='float')

    #Translate oneHot results 
    for j in range(oneHotRegistry.shape[0]):
        data[j] = translateOneHot(oneHotRegistry[j,:])


    #Create id labels
    labels = range(data.shape[0]+1)
    labels = labels[1:]

   
    #Concat data and labels
    toFileData = numpy.stack((labels,data),axis=-1)
    toFileData = toFileData.astype(numpy.int32)


    #Save new results

    numpy.savetxt('cnn_output_translated.csv',toFileData,header="ImageId,Label",delimiter=',',fmt='%i')

