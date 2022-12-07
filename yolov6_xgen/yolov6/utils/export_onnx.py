from yolov6.models.yolo import *
from yolov6.models.effidehead import Detect
from yolov6.layers.common import *
from yolov6.utils.events import LOGGER
from yolov6.utils.checkpoint import load_checkpoint
from io import BytesIO
import torch
from yolov6.utils.torch_utils import fuse_model
import onnx
import os

from xgen_tools.args_define import XgenArgs

def export_onnx(args_ai, model):
    input_shape = [1, 3, 640, 640]
    onnx_save_path = os.path.join(args_ai[XgenArgs.cocopie_general][XgenArgs.cocopie_work_place],
                                  args_ai[XgenArgs.cocopie_train][XgenArgs.cocopie_uuid] + '.onnx')
    model = fuse_model(model).eval()
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()
    # Input
    img = torch.zeros(1, 3, 640, 640).to("cuda")  # image size(1,3,320,192) iDetection

    # Update model
    # if args.half:
    #     img, model = img.half(), model.half()  # to FP16
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = False
    dynamic_axes = None
    # if args.dynamic_batch:
    #     args.batch_size = 'batch'
    #     dynamic_axes = {
    #         'images': {
    #             0: 'batch',
    #         }, }
    #     if args.end2end:
    #         output_axes = {
    #             'num_dets': {0: 'batch'},
    #             'det_boxes': {0: 'batch'},
    #             'det_scores': {0: 'batch'},
    #             'det_classes': {0: 'batch'},
    #         }
    #     else:
    #         output_axes = {
    #             'outputs': {0: 'batch'},
    #         }
    #     dynamic_axes.update(output_axes)

    # if args.end2end:
    #     from yolov6.models.end2end import End2End
    #
    #     model = End2End(model, max_obj=args.topk_all, iou_thres=args.iou_thres, score_thres=args.conf_thres,
    #                     device=device, ort=args.ort, trt_version=args.trt_version, with_preprocess=args.with_preprocess)

    print("===================")
    print(model)
    print("===================")

    y = model(img)  # dry run

    # ONNX export
    try:
        LOGGER.info('\nStarting to export ONNX...')
        # export_file = args.weights.replace('.pt', '_test.onnx')  # filename
        with BytesIO() as f:
            torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                              training=torch.onnx.TrainingMode.EVAL,
                              do_constant_folding=True,
                              input_names=['input'],
                              output_names=['output'],
                              dynamic_axes=dynamic_axes)
            f.seek(0)
            # Checks
            onnx_model = onnx.load(f)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model
            # Fix output shape
            # if args.end2end and not args.ort:
            #     shapes = [args.batch_size, 1, args.batch_size, args.topk_all, 4,
            #               args.batch_size, args.topk_all, args.batch_size, args.topk_all]
            #     for i in onnx_model.graph.output:
            #         for j in i.type.tensor_type.shape.dim:
            #             j.dim_param = str(shapes.pop(0))
        if True:
            try:
                import onnxsim

                LOGGER.info('\nStarting to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
            except Exception as e:
                LOGGER.info(f'Simplifier failure: {e}')
        onnx.save(onnx_model, onnx_save_path)
        LOGGER.info(f'ONNX export success, saved as {onnx_save_path}')
    except Exception as e:
        LOGGER.info(f'ONNX export failure: {e}')

    # Finish
    # LOGGER.info('\nExport complete (%.2fs)' % (time.time() - t))
    # if args.end2end:
    #     if not args.ort:
    #         info = f'trtexec --onnx={export_file} --saveEngine={export_file.replace(".onnx", ".engine")}'
    #         if args.dynamic_batch:
    #             LOGGER.info('Dynamic batch export should define min/opt/max batchsize\n' +
    #                         'We set min/opt/max = 1/16/32 default!')
    #             wandh = 'x'.join(list(map(str, args.img_size)))
    #             info += (f' --minShapes=images:1x3x{wandh}' +
    #                      f' --optShapes=images:16x3x{wandh}' +
    #                      f' --maxShapes=images:32x3x{wandh}' +
    #                      f' --shapes=images:16x3x{wandh}')
    #         LOGGER.info('\nYou can export tensorrt engine use trtexec tools.\nCommand is:')
    #         LOGGER.info(info)

