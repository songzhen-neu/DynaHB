import argparse
from adgnn.context import context


def parserInit():
    parser = argparse.ArgumentParser(description="Pytorch argument parser")
    parser.add_argument('--id', type=str, help='id')
    parser.add_argument('--ifctx', type=str, help='context from file or arguments delivering')
    parser.add_argument('--worker_num', type=str, help='the number of worker and server')
    parser.add_argument('--vtx_feat_class_ratio', type=str, help='vtx_feat_class_ratio')

    parser.add_argument('--hidden', type=str, help='hidden')
    parser.add_argument('--data_path', type=str, help='data_path')

    parser.add_argument('--iter_lr_pttMethod_printInterval_device', type=str, help='iter_lr_pttMethod')
    parser.add_argument('--workers', type=str, help='worker ip')


    args = parser.parse_args()
    ifctx = args.ifctx

    if ifctx == 'true':
        context.Context.ifctx = True
    else:
        context.Context.ifctx = False

    if context.Context.ifctx:
        print("using context configuration")

        context.glContext.config['id'] = int(args.id)

        context.glContext.config['layer_num'] = len(context.glContext.config['hidden'])
        context.glContext.config['emb_dims'].append(context.glContext.config['feature_dim'])
        context.glContext.config['emb_dims'].extend(context.glContext.config['hidden'])
        context.glContext.config['emb_dims'].append(context.glContext.config['class_num'])
    else:
        print("using argument configuration")
        context.glContext.config['id'] = int(args.id)

        context.glContext.config['worker_num'] = int(args.worker_num)


        context.glContext.config['data_path'] = args.data_path

        context.glContext.config['hidden'] = [int(i) for i in str.split(args.hidden, ':')]
        context.glContext.config['layer_num'] = len(context.glContext.config['hidden'])


        vtx_feat_class_ratio = str.split(args.vtx_feat_class_ratio, ',')
        context.glContext.config['data_num'] = int(vtx_feat_class_ratio[0])
        context.glContext.config['feature_dim'] = int(vtx_feat_class_ratio[1])
        context.glContext.config['class_num'] = int(vtx_feat_class_ratio[2])
        context.glContext.config['train_num'] = float(vtx_feat_class_ratio[3])


        context.glContext.config['emb_dims'].append(context.glContext.config['feature_dim'])
        context.glContext.config['emb_dims'].extend(context.glContext.config['hidden'])
        context.glContext.config['emb_dims'].append(context.glContext.config['class_num'])

        iter_lr_pttMethod_printInterval_device= str.split(args.iter_lr_pttMethod_printInterval_device, ',')
        context.glContext.config['iterNum'] = int(iter_lr_pttMethod_printInterval_device[0])
        context.glContext.config['lr'] = float(iter_lr_pttMethod_printInterval_device[1])
        context.glContext.config['partitionMethod'] = iter_lr_pttMethod_printInterval_device[2]
        context.glContext.config['print_result_interval'] = int(iter_lr_pttMethod_printInterval_device[3])
        context.glContext.config['device'] = iter_lr_pttMethod_printInterval_device[4]


    context.glContext.ipInit(args.workers)
    printContext()


def printContext():
    print("*************************context info****************************")
    for id in context.glContext.config:
        print("{0} = {1}".format(id,context.glContext.config[id]))
    print("*************************context info****************************")
