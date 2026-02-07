import sys
print(f"Python {sys.version}")

# 测试导入
packages = ['numpy', 'scipy', 'matplotlib', 'cv2', 'pywt', 'skimage']
for pkg in packages:
    try:
        __import__(pkg)
        version = __import__(pkg).__version__
        print(f"✓ {pkg}: {version}")
    except ImportError:
        print(f"✗ {pkg}: 未安装")
    except AttributeError:
        print(f"✓ {pkg}: 已安装")

# 测试BM3D
try:
    import bm3d
    print("✓ bm3d: 已安装")
    
    # 简单测试
    import numpy as np
    test_img = np.random.rand(64, 64).astype(np.float32)
    denoised = bm3d.bm3d(test_img, sigma_psd=25)
    print("✓ bm3d: 功能正常")
except ImportError:
    print("✗ bm3d: 未安装 - 请从 https://www.lfd.uci.edu/~gohlke/pythonlibs/#bm3d 下载whl文件")
except Exception as e:
    print(f"✗ bm3d: 错误 - {e}")