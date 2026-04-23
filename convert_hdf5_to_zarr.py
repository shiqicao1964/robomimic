# convert_real_data_to_zarr.py
import h5py
import numpy as np
import zarr
from tqdm import tqdm
import os
import argparse
from numcodecs import VLenUTF8, Zlib

def convert_real_robomimic_to_zarr(
    hdf5_path,
    output_zarr_path,
    task_name="real_pick",
    prompt_file=None
):
    """
    将 real 数据集的 HDF5 文件转换为 Zarr 格式
    适配 demo.hdf5 的结构到项目配置
    """
    print(f"Loading data from: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        demos = list(f['data'].keys())
        print(f"Found {len(demos)} demonstrations")
        
        # 计算总帧数和 episode 边界
        episode_ends = []
        total_steps = 0
        
        for demo in demos:
            demo_len = f[f'data/{demo}/actions'].shape[0]
            total_steps += demo_len
            episode_ends.append(total_steps)
        
        print(f"Total steps: {total_steps}")
        
        # 创建 Zarr 结构
        root = zarr.group(output_zarr_path, overwrite=True)
        data_group = root.require_group('data', overwrite=True)
        meta_group = root.require_group('meta', overwrite=True)
        obs_group = data_group.require_group('obs', overwrite=True)
        lowdim_group = obs_group.require_group('lowdim', overwrite=True)
        rgb_group = obs_group.require_group('rgb', overwrite=True)
        action_group = data_group.require_group('action', overwrite=True)
        
        # 保存 episode_ends
        meta_group.array('episode_ends', episode_ends, dtype=np.int64, compressor=None)
        
        # 读取 prompt
        if prompt_file and os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as pf:
                prompt_text = pf.read().strip()
        else:
            prompt_text = f"Pick up the {task_name} object"
        
        # 修复：使用 VLenUTF8 编解码器对象
        root.array('prompt', np.array([prompt_text], dtype=object), dtype=object,
                   object_codec=VLenUTF8(), compressor=None)
        
        # 处理低维数据: 从 ee_pose (7维) 拆分为 eef_pos (3维) 和 eef_angle (4维)
        print("\nProcessing low-dim state data...")
        all_eef_pos = []
        all_eef_angle = []
        all_gripper = []
        
        for demo in tqdm(demos, desc="States"):
            demo_len = f[f'data/{demo}/actions'].shape[0]
            ee_pose = f[f'data/{demo}/obs/ee_pose'][:demo_len]  # shape: (N, 7)
            gripper = f[f'data/{demo}/obs/gripper_position'][:demo_len]  # shape: (N, 1)
            
            # 拆分 ee_pose: 前3维位置，后4维四元数
            eef_pos = ee_pose[:, :3]      # (N, 3)
            eef_angle = ee_pose[:, 3:7]   # (N, 4)
            
            all_eef_pos.append(eef_pos)
            all_eef_angle.append(eef_angle)
            all_gripper.append(gripper)
        
        # 拼接并保存低维数据
        eef_pos_data = np.concatenate(all_eef_pos, axis=0)
        eef_angle_data = np.concatenate(all_eef_angle, axis=0)
        gripper_data = np.concatenate(all_gripper, axis=0)
        
        print(f"  eef_pos shape: {eef_pos_data.shape}")
        print(f"  eef_angle shape: {eef_angle_data.shape}")
        print(f"  gripper shape: {gripper_data.shape}")
        
        # 按配置的键名保存低维数据
        lowdim_group.array('eef_pos_x', eef_pos_data[:, 0:1], compressor=None, dtype=np.float32)
        lowdim_group.array('eef_pos_y', eef_pos_data[:, 1:2], compressor=None, dtype=np.float32)
        lowdim_group.array('eef_pos_z', eef_pos_data[:, 2:3], compressor=None, dtype=np.float32)
        lowdim_group.array('eef_angle_x', eef_angle_data[:, 0:1], compressor=None, dtype=np.float32)
        lowdim_group.array('eef_angle_y', eef_angle_data[:, 1:2], compressor=None, dtype=np.float32)
        lowdim_group.array('eef_angle_z', eef_angle_data[:, 2:3], compressor=None, dtype=np.float32)
        lowdim_group.array('eef_angle_w', eef_angle_data[:, 3:4], compressor=None, dtype=np.float32)
        lowdim_group.array('gripper_open', gripper_data, compressor=None, dtype=np.float32)
        
        # 处理动作数据
        print("\nProcessing actions...")
        all_actions = []
        for demo in tqdm(demos, desc="Actions"):
            demo_len = f[f'data/{demo}/actions'].shape[0]
            actions = f[f'data/{demo}/actions'][:demo_len]
            all_actions.append(actions)
        
        actions_data = np.concatenate(all_actions, axis=0)
        print(f"  actions shape: {actions_data.shape}")
        
        # 保存为配置期望的格式
        action_keys = ['eef_pos_x', 'eef_pos_y', 'eef_pos_z',
                       'eef_angle_roll', 'eef_angle_pitch', 'eef_angle_yaw',
                       'gripper_open']
        
        for i, key in enumerate(action_keys):
            if i < actions_data.shape[1]:
                action_group.array(key, actions_data[:, i:i+1], 
                                 compressor=None, dtype=np.float32)
                print(f"    {key}: {actions_data[:, i:i+1].shape}")
        
        # 处理图像数据
        print("\nProcessing images...")
        for img_key in ['image', 'image_wrist']:
            all_images = []
            for demo in tqdm(demos, desc=f"Images - {img_key}"):
                demo_len = f[f'data/{demo}/actions'].shape[0]
                images = f[f'data/{demo}/obs/{img_key}'][:demo_len]
                all_images.append(images)
            
            if all_images:
                images_data = np.concatenate(all_images, axis=0)
                print(f"  {img_key} shape: {images_data.shape}")
                # 使用配置中的键名，添加压缩
                rgb_group.array(img_key, images_data,
                              chunks=(1, images_data.shape[1], images_data.shape[2], images_data.shape[3]),
                              compressor=Zlib(level=5), dtype=np.uint8)
    
    print(f"\n Conversion complete!")
    print(f"   Output: {output_zarr_path}")
    
    # 验证输出
    print("\nVerifying output...")
    root = zarr.open(output_zarr_path, mode='r')
    print(f"   Groups: {list(root.keys())}")
    if 'data/obs/lowdim' in root:
        lowdim_group = root['data/obs/lowdim']
        print(f"   Low-dim keys: {list(lowdim_group.keys())}")
    if 'data/obs/rgb' in root:
        rgb_group = root['data/obs/rgb']
        print(f"   RGB keys: {list(rgb_group.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_path', type=str, required=True,
                        help='Path to demo.hdf5 file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--task_name', type=str, default='real_pick',
                        help='Task name')
    parser.add_argument('--prompt_file', type=str, default=None,
                        help='Path to prompt text file')
    args = parser.parse_args()
    
    output_zarr_path = os.path.join(args.output_dir, args.task_name, 'dataset_buffer.zarr')
    os.makedirs(os.path.dirname(output_zarr_path), exist_ok=True)
    
    convert_real_robomimic_to_zarr(
        hdf5_path=args.hdf5_path,
        output_zarr_path=output_zarr_path,
        task_name=args.task_name,
        prompt_file=args.prompt_file
    )
