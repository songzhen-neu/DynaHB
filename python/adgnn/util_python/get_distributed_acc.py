import adgnn.context.context as context
def getAccAvrg(split_vnum, split_acc):
    acc_avrg = {}
    acc_entire = context.glContext.dgnnWorkerRouter[0].sendAccuracy(split_vnum,split_acc)
    context.glContext.dgnnWorkerRouter[0].barrier()
    acc_avrg['train']=acc_entire[0]
    acc_avrg['test']=acc_entire[1]
    return acc_avrg