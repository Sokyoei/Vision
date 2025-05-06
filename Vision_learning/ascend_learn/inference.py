import numpy as np
from ascend import acl


def main():
    # 初始化ACL资源
    acl.init()

    # 创建上下文
    context, stream = acl.rt.set_device(0)

    # 加载OM模型
    model = acl.model.load_model('yolov8_pose.om')

    # 创建输入和输出内存
    input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
    input_buffer, input_size = acl.util.numpy_to_buffer(input_data)
    output_buffer, output_size = acl.util.create_buffer(model.output_size)

    # 推理
    acl.model.execute(model, [input_buffer], [output_buffer])

    # 处理输出结果
    output_data = acl.util.buffer_to_numpy(output_buffer, output_size)
    print(output_data)

    # 释放资源
    acl.model.unload_model(model)
    acl.rt.destroy_context(context)
    acl.finalize()


if __name__ == "__main__":
    main()
