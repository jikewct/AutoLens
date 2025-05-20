"""
"Hello. world!" for DeepLens.

In this code, we will load a lens from a file. Then we will plot the lens setup and render a sample image.
"""

import random
import string
from datetime import datetime

import yaml
from deeplens import (
    DEFAULT_WAVE,
    DEPTH,
    EPSILON,
    WAVE_RGB,
    GeoLens,
    create_camera_lens,
    create_cellphone_lens,
    create_video_from_images,
    set_logger,
    set_seed,
)
from deeplens.optics import *
from monitor import *
from transformers import get_cosine_schedule_with_warmup


def config():
    """Config file for training."""
    # Config file
    with open("configs/autolens.yml") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # Result dir
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for i in range(4))
    current_time = datetime.now().strftime("%m%d-%H%M%S")
    exp_name = current_time + "-AutoLens-RMS-" + random_string
    result_dir = f"./results/{exp_name}"
    os.makedirs(result_dir, exist_ok=True)
    args["result_dir"] = result_dir

    if args["seed"] is None:
        seed = random.randint(0, 100)
        args["seed"] = seed
    set_seed(args["seed"])

    # Log
    set_logger(result_dir)
    logging.info(f'EXP: {args["EXP_NAME"]}')

    # Device
    num_gpus = torch.cuda.device_count()
    args["num_gpus"] = num_gpus
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    args["device"] = device
    logging.info(f"Using {num_gpus} {torch.cuda.get_device_name(0)} GPU(s)")

    return args


def init_from_file():
    args = config()
    result_dir = args["result_dir"]
    device = args["device"]
    # device = torch.device("cpu")
    torch.set_default_device(device)
    lens = GeoLens(filename="/home/wangchangtao/Data/Code/AutoLens/configs/iter300.json")
    depth = -20000.0
    num_grid = 21
    spp = 512
    iterations = 5000
    i = 5000
    aper_start = lens.surfaces[lens.aper_idx].r * 0.5
    aper_final = lens.surfaces[lens.aper_idx].r
    lrs = [5e-4, 1e-4, 0.1, 1e-4]
    decay = 0.02
    optimizer = lens.get_optimizer(lrs, decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=iterations // 10, num_training_steps=iterations)

    aper_r = min((aper_final - aper_start) * (i / iterations * 1.1) + aper_start, aper_final)
    lens.surfaces[lens.aper_idx].r = aper_r
    lens.fnum = lens.foclen / aper_r / 2
    rays_backup: list[Ray] = []
    for wv in WAVE_RGB:
        scale = lens.calc_scale_pinhole(depth)
        # rays_backup.clear()
        ray = lens.sample_point_source(
            M=num_grid,
            R=lens.sensor_size[0] / 2 * scale,
            depth=depth,
            spp=spp,
            pupil=True,
            wvln=wv,
            importance_sampling=True,
        )
        ray.propagate_to(lens.surfaces[0].d - 10.0)
        rays_backup.append(ray)

    ray = rays_backup[0].clone()
    return lens, ray, rays_backup


def init():
    args = config()
    result_dir = args["result_dir"]
    device = args["device"]
    # device = torch.device("cpu")
    torch.set_default_device(device)
    # ===> Create a cellphone lens
    lens = create_cellphone_lens(hfov=args["HFOV"], imgh=args["DIAG"], fnum=args["FNUM"], lens_num=args["lens_num"], save_dir=result_dir)
    lens.set_target_fov_fnum(hfov=args["HFOV"], fnum=args["FNUM"], imgh=args["DIAG"])
    logging.info(
        f'==> Design target: FOV {round(args["HFOV"]*2*57.3, 2)}, DIAG {args["DIAG"]}mm, F/{args["FNUM"]}, FOCLEN {round(args["DIAG"]/2/np.tan(args["HFOV"]), 2)}mm.'
    )
    depth = -20000.0
    num_grid = 21
    spp = 512
    iterations = 5000
    i = 5000
    aper_start = lens.surfaces[lens.aper_idx].r * 0.5
    aper_final = lens.surfaces[lens.aper_idx].r
    lrs = [5e-4, 1e-4, 0.1, 1e-4]
    decay = 0.02
    optimizer = lens.get_optimizer(lrs, decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=iterations // 10, num_training_steps=iterations)

    aper_r = min((aper_final - aper_start) * (i / iterations * 1.1) + aper_start, aper_final)
    lens.surfaces[lens.aper_idx].r = aper_r
    lens.fnum = lens.foclen / aper_r / 2
    rays_backup: list[Ray] = []
    for wv in WAVE_RGB:
        scale = lens.calc_scale_pinhole(depth)
        # rays_backup.clear()
        ray = lens.sample_point_source(
            M=num_grid,
            R=lens.sensor_size[0] / 2 * scale,
            depth=depth,
            spp=spp,
            pupil=True,
            wvln=wv,
            importance_sampling=True,
        )
        ray.propagate_to(lens.surfaces[0].d - 10.0)
        rays_backup.append(ray)

    ray = rays_backup[0].clone()
    return lens, ray, rays_backup


def test_surface_1_fwd(lens, ray):
    mat1 = lens.surfaces[0].mat2
    n1 = mat1.ior(ray.wvln)
    n2 = lens.surfaces[1].mat2.ior(ray.wvln)
    print(ray.o.shape, ray.o.device)
    time_cost = TimeMeter()
    for j in range(3):
        time_cost.start()
        for i in range(1000):
            tmp_ray1 = ray.clone()
            t, valid = lens.surfaces[1].newtons_method(tmp_ray1)
        print(t.shape)
        print("0 time cost:", time_cost.intervalToNow())
        time_cost.reset()
        time_cost.start()
        for i in range(1000):
            tmp_ray2 = ray.clone()
            t_2, valid_2 = lens.surfaces[1].newtons_method2(tmp_ray2)
        print(
            "max diff",
            torch.max(torch.abs(t - t_2)),
            torch.max(torch.abs(valid.to(torch.float32) - valid_2.to(torch.float32))),
        )
        print("triton time cost:", time_cost.intervalToNow())
        # mat1 = lens.surfaces[1].mat2
        # n1 = mat1.ior(ray.wvln)
        # n2 = lens.surfaces[1].mat2.ior(ray.wvln)
        # time_cost.start()
        # for i in range(1):
        #     tmp_ray = t.clone()
        #     t = lens.surfaces[1].ray_reaction(tmp_ray, n1, n2)
        # print("1 time cost:", time_cost.intervalToNow())
        time_cost.reset()


def test_ray_intersection(lens, ray):
    mat1 = lens.surfaces[0].mat2
    n1 = mat1.ior(ray.wvln)
    clen = lens.surfaces[1]
    nclen = clen.clone()
    n2 = clen.mat2.ior(ray.wvln)
    print(ray.o.shape, ray.o.device)
    # unitTest(clen, ray)
    time_cost = TimeMeter()
    iters = 1
    for j in range(1):

        for i in range(iters):
            time_cost.start()
            tmp_ray1 = ray.clone()
            # tmp_ray1.o.requires_grad = True
            # tmp_ray1.d.requires_grad = True
            # tmp_ray1.ra.requires_grad = True
            ray_result = clen.intersect(tmp_ray1, n1 / n2)
            # t.sum().backward()

        print("0 time cost:", time_cost.intervalToNow())
        time_cost.reset()

        time_cost.start()
        for i in range(iters):
            tmp_ray2 = ray.clone()
            # tmp_ray2.o.requires_grad = True
            # tmp_ray2.d.requires_grad = True
            # tmp_ray2.ra.requires_grad = True
            ray_result2 = nclen.newtons_method2(ray)

        print("d diff:", (ray_result.d - ray_result2.d).abs().max())
        print("o diff:", (ray_result.o - ray_result2.o).abs().max())
        print("ra diff:", (ray_result.ra - ray_result2.ra).abs().max())
        print("obliq diff:", (ray_result.obliq - ray_result2.obliq).abs().max())
        print(ray_result.o[0, :2, :2, :], ray_result2.o[0, :2, :2, :])
        print("triton time cost:", time_cost.intervalToNow())


def test_ray_intersection_bwd(lens, ray):
    mat1 = lens.surfaces[0].mat2
    n1 = mat1.ior(ray.wvln)
    clen = lens.surfaces[1]
    nclen = clen.clone()
    n2 = clen.mat2.ior(ray.wvln)
    print(ray.o.shape, ray.o.device)
    # unitTest(clen, ray)
    time_cost = TimeMeter()
    iters = 1
    for j in range(1):
        time_cost.start()
        for i in range(iters):
            tmp_ray1 = ray.clone()
            tmp_ray1.o.requires_grad = True
            tmp_ray1.d.requires_grad = True
            tmp_ray1.ra.requires_grad = True
            tmp1_o, tmp1_d, tmp1_ra, tmp1_obliq = tmp_ray1.o, tmp_ray1.d, tmp_ray1.ra, tmp_ray1.obliq
            ray_result = clen.intersect(tmp_ray1, n1 / n2)
            sum1 = ray_result.d.sum() + ray_result.o.sum() + ray_result.ra.sum() + ray_result.obliq.sum()
            sum1.backward()

        print("0 time cost:", time_cost.intervalToNow())
        time_cost.reset()

        time_cost.start()
        for i in range(iters):
            tmp_ray2 = ray.clone()
            tmp_ray2.o.requires_grad = True
            tmp_ray2.d.requires_grad = True
            tmp_ray2.ra.requires_grad = True
            tmp2_o, tmp2_d, tmp2_ra, tmp2_obliq = tmp_ray2.o, tmp_ray2.d, tmp_ray2.ra, tmp_ray2.obliq

            ray_result2 = nclen.newtons_method2(tmp_ray2)
            sum2 = ray_result2.d.sum() + ray_result2.o.sum() + ray_result2.ra.sum() + ray_result2.obliq.sum()
            sum2.backward()
            #     # t_2.sum().backward()
            # print(ray_result.d[0, :2, :2, :], ray_result2.d[0, :2, :2, :])
        print("triton time cost:", time_cost.intervalToNow())
    print("d diff:", (ray_result.d - ray_result2.d).abs().max())
    print("o diff:", (ray_result.o - ray_result2.o).abs().max())
    print("ra diff:", (ray_result.ra - ray_result2.ra).abs().max())
    print("obliq diff:", (ray_result.obliq - ray_result2.obliq).abs().max())
    print("o grad diff:", (tmp1_o.grad - tmp2_o.grad).abs().max())
    print("d grad diff:", (tmp1_d.grad - tmp2_d.grad).abs().max())
    print("ra grad diff:", (tmp1_ra.grad - tmp2_ra.grad).abs().max())
    print("obliq grad diff:", tmp1_obliq.grad, tmp2_obliq.grad)
    print("d grad diff:", clen.d.grad, nclen.d.grad)
    print("c grad diff:", clen.c.grad, nclen.c.grad)
    print("ai2 grad diff:", clen.ai2.grad, nclen.ai2.grad)
    print("ai4 grad diff:", clen.ai4.grad, nclen.ai4.grad)
    print("ai6 grad diff:", clen.ai6.grad, nclen.ai6.grad)
    print("ai8 grad diff:", clen.ai8.grad, nclen.ai8.grad)
    print("ai10 grad diff:", clen.ai10.grad, nclen.ai10.grad)
    print("ai12 grad diff:", clen.ai12.grad, nclen.ai12.grad)
    print(clen.ai2.dtype)
    # print("o grad diff:",


def test_surface_1_bwd(lens, ray):
    mat1 = lens.surfaces[0].mat2
    n1 = mat1.ior(ray.wvln)
    clen = lens.surfaces[1]
    nclen = clen.clone()
    n2 = clen.mat2.ior(ray.wvln)
    print(ray.o.shape, ray.o.device)
    unitTest(clen, ray)
    time_cost = TimeMeter()
    iters = 100
    for j in range(3):
        tmp_ray1 = ray.clone()
        tmp_ray1.o.requires_grad = True
        tmp_ray1.d.requires_grad = True
        tmp_ray1.ra.requires_grad = True
        tmp_ray2 = ray.clone()
        tmp_ray2.o.requires_grad = True
        tmp_ray2.d.requires_grad = True
        tmp_ray2.ra.requires_grad = True
        time_cost.start()
        for i in range(iters):

            t, valid = clen.newtons_method(tmp_ray1)
            t.sum().backward()

        print("0 time cost:", time_cost.intervalToNow())
        time_cost.reset()

        time_cost.start()
        for i in range(iters):

            t_2, valid_2 = nclen.newtons_method2(tmp_ray2)
            t_2.sum().backward()
        print("triton time cost:", time_cost.intervalToNow())


def test_ray_refraction(lens, ray):
    mat1 = lens.surfaces[0].mat2
    n1 = mat1.ior(ray.wvln)
    clen = lens.surfaces[1]
    nclen = clen.clone()
    n2 = clen.mat2.ior(ray.wvln)
    print(ray.o.shape, ray.o.device)
    # unitTest(clen, ray)
    time_cost = TimeMeter()
    iters = 1
    is_ray1 = clen.intersect(ray.clone(), n1)
    is_ray2 = nclen.intersect(ray.clone(), n1)
    for j in range(1):

        for i in range(iters):
            time_cost.start()
            # tmp_ray1 = is_ray1.clone()
            # tmp_ray1.o.requires_grad = True
            # tmp_ray1.d.requires_grad = True
            # tmp_ray1.ra.requires_grad = True
            ray_result = clen.refract(is_ray1, n1 / n2)
            # t.sum().backward()

        print("0 time cost:", time_cost.intervalToNow())
        time_cost.reset()

        time_cost.start()
        for i in range(iters):
            # tmp_ray2 = is_ray2.clone()
            # tmp_ray2.o.requires_grad = True
            # tmp_ray2.d.requires_grad = True
            # tmp_ray2.ra.requires_grad = True
            ray_result2 = nclen.refract_new(is_ray2, n1 / n2)

        print("d diff:", (ray_result.d - ray_result2.d).abs().max())
        print("o diff:", (ray_result.o - ray_result2.o).abs().max())
        print("ra diff:", (ray_result.ra - ray_result2.ra).abs().max())
        print("obliq diff:", (ray_result.obliq - ray_result2.obliq).abs().max())
        print(ray_result.d[0, :2, :2, :], ray_result2.d[0, :2, :2, :])
        print("triton time cost:", time_cost.intervalToNow())


def test_ray_reaction(lens, ray):
    mat1 = lens.surfaces[0].mat2
    n1 = mat1.ior(ray.wvln)
    clen = lens.surfaces[1]
    nclen = clen.clone()
    n2 = clen.mat2.ior(ray.wvln)
    print(ray.o.shape, ray.o.device)
    # unitTest(clen, ray)
    time_cost = TimeMeter()
    iters = 100
    for j in range(5):
        time_cost.start()
        for i in range(iters):
            tmp_ray1 = ray.clone()
            tmp_ray1.o.requires_grad = True
            tmp_ray1.d.requires_grad = True
            tmp_ray1.ra.requires_grad = True
            ray_result = clen.ray_reaction(tmp_ray1, n1, n2, method="old")
            # t.sum().backward()

        print("0 time cost:", time_cost.intervalToNow())
        time_cost.reset()

        time_cost.start()
        for i in range(iters):
            tmp_ray2 = ray.clone()
            tmp_ray2.o.requires_grad = True
            tmp_ray2.d.requires_grad = True
            tmp_ray2.ra.requires_grad = True
            ray_result2 = nclen.ray_reaction(tmp_ray2, n1, n2, method="new")

        #     # t_2.sum().backward()
        print("triton time cost:", time_cost.intervalToNow())
    print("d diff:", (ray_result.d - ray_result2.d).abs().max())
    print("o diff:", (ray_result.o - ray_result2.o).abs().max())
    print("ra diff:", (ray_result.ra - ray_result2.ra).abs().max())
    print("obliq diff:", (ray_result.obliq - ray_result2.obliq).abs().max())


def test_ray_reaction_bwd(lens, ray):
    mat1 = lens.surfaces[0].mat2
    n1 = mat1.ior(ray.wvln)
    clen = lens.surfaces[1]
    nclen = clen.clone()
    n2 = clen.mat2.ior(ray.wvln)
    print(ray.o.shape, ray.o.device)
    # unitTest(clen, ray)
    time_cost = TimeMeter()
    iters = 100
    for j in range(5):
        time_cost.start()
        for i in range(iters):
            tmp_ray1 = ray.clone()
            tmp_ray1.o.requires_grad = True
            tmp_ray1.d.requires_grad = True
            tmp_ray1.ra.requires_grad = True
            ray_result = clen.ray_reaction(tmp_ray1, n1, n2, method="old", refactor="test")

            (ray_result.d.sum() + ray_result.o.sum()).backward()

        print("0 time cost:", time_cost.intervalToNow())
        time_cost.reset()

        time_cost.start()
        for i in range(iters):
            tmp_ray2 = ray.clone()
            tmp_ray2.o.requires_grad = True
            tmp_ray2.d.requires_grad = True
            tmp_ray2.ra.requires_grad = True
            ray_result2 = nclen.ray_reaction(tmp_ray2, n1, n2, method="new", refactor="new")
            (ray_result2.d.sum() + +ray_result2.o.sum()).backward()
        #     # t_2.sum().backward()
        print("triton time cost:", time_cost.intervalToNow())
        # print("d diff:", (ray_result.d - ray_result2.d).abs().max())
        # print("o diff:", (ray_result.o - ray_result2.o).abs().max())
        # print("ra diff:", (ray_result.ra - ray_result2.ra).abs().max())
        # print("obliq diff:", (ray_result.obliq - ray_result2.obliq).abs().max())
        # print("c grad diff:", clen.c.grad - nclen.c.grad)
        # print("ai2 grad diff:", clen.ai2.grad - nclen.ai2.grad)
        # print("ai4 grad diff:", clen.ai4.grad - nclen.ai4.grad)
        # print("ai6 grad diff:", clen.ai6.grad - nclen.ai6.grad)
        # print("ai8 grad diff:", clen.ai8.grad - nclen.ai8.grad)
        # print("ai10 grad diff:", clen.ai10.grad - nclen.ai10.grad)
        # print("ai12 grad diff:", clen.ai12.grad - nclen.ai12.grad)
        # print(clen.ai2.dtype)
        # print("o grad diff:", (tmp_ray1.d.grad - tmp_ray2.d.grad).abs().max())


def test_ray_refraction_bwd(lens, ray):
    mat1 = lens.surfaces[0].mat2
    n1 = mat1.ior(ray.wvln)
    clen = lens.surfaces[1]
    nclen = clen.clone()
    n2 = clen.mat2.ior(ray.wvln)
    print(ray.o.shape, ray.o.device)
    ray.o.requires_grad = True
    ray.d.requires_grad = True
    ray.ra.requires_grad = True
    ray.obliq.requires_grad = True
    ray1 = clen.intersect(ray.clone(), n1 / n2)
    ray2 = nclen.intersect(ray.clone(), n1 / n2)
    print("intersect d diff:", (ray1.d - ray2.d).abs().max())
    print("intersect o diff:", (ray1.o - ray2.o).abs().max())
    print("intersectr a diff:", (ray1.ra - ray2.ra).abs().max())
    print("intersect obliq diff:", (ray1.obliq - ray2.obliq).abs().max())
    # unitTest(clen, ray)
    time_cost = TimeMeter()
    iters = 1
    for j in range(1):
        time_cost.start()
        for i in range(iters):
            tmp_ray1 = ray1
            # tmp_ray1.o.requires_grad = True
            # tmp_ray1.d.requires_grad = True
            # tmp_ray1.ra.requires_grad = True
            ray_result = clen.refract_test(tmp_ray1, n1 / n2)
            sum1 = ray_result.d.sum() + ray_result.o.sum() + ray_result.ra.sum() + ray_result.obliq.sum()
            sum1.backward()

        print("0 time cost:", time_cost.intervalToNow())
        time_cost.reset()

        time_cost.start()
        for i in range(iters):
            tmp_ray2 = ray2
            # tmp_ray2.o.requires_grad = True
            # tmp_ray2.d.requires_grad = True
            # tmp_ray2.ra.requires_grad = True
            ray_result2 = nclen.refract_new(tmp_ray2, n1 / n2)
            sum2 = ray_result2.d.sum() + ray_result2.o.sum() + ray_result2.ra.sum() + ray_result2.obliq.sum()
            sum2.backward()
            #     # t_2.sum().backward()
            # print(ray_result.d[0, :2, :2, :], ray_result2.d[0, :2, :2, :])
        print("triton time cost:", time_cost.intervalToNow())
    print("d diff:", (ray_result.d - ray_result2.d).abs().max())
    print("o diff:", (ray_result.o - ray_result2.o).abs().max())
    print("ra diff:", (ray_result.ra - ray_result2.ra).abs().max())
    print("obliq diff:", (ray_result.obliq - ray_result2.obliq).abs().max())
    print("c grad diff:", clen.c.grad, nclen.c.grad)
    print("d grad diff:", clen.d.grad, nclen.d.grad)
    print("ai2 grad diff:", clen.ai2.grad, nclen.ai2.grad)
    print("ai4 grad diff:", clen.ai4.grad, nclen.ai4.grad)
    print("ai6 grad diff:", clen.ai6.grad, nclen.ai6.grad)
    print("ai8 grad diff:", clen.ai8.grad, nclen.ai8.grad)
    print("ai10 grad diff:", clen.ai10.grad, nclen.ai10.grad)
    print("ai12 grad diff:", clen.ai12.grad, nclen.ai12.grad)
    print(clen.ai2.dtype)
    # print("o grad diff:",


def test_surface_sag(lens, ray):
    mat1 = lens.surfaces[0].mat2
    n1 = mat1.ior(ray.wvln)
    clen = lens.surfaces[1]
    nclen = clen.clone()
    n2 = clen.mat2.ior(ray.wvln)
    print(ray.o.shape, ray.o.device)
    # unitTest(clen, ray)
    time_cost = TimeMeter()
    iters = 1
    for j in range(1):
        tmp_ray1 = ray.clone()
        tmp_ray1.o.requires_grad = True
        tmp_ray1.d.requires_grad = True
        tmp_ray1.ra.requires_grad = True
        tmp_ray2 = ray.clone()
        tmp_ray2.o.requires_grad = True
        tmp_ray2.d.requires_grad = True
        tmp_ray2.ra.requires_grad = True
        time_cost.start()
        for i in range(iters):
            
            z = clen.sag_test(tmp_ray1.o[..., 0], tmp_ray1.o[..., 1])
            # z.sum().backward()
        print("0 time cost:", time_cost.intervalToNow())
        time_cost.reset()

        time_cost.start()
        for i in range(iters):

            z2 = nclen.surface(tmp_ray2.o[..., 0], tmp_ray2.o[..., 1], use_flash_method=True)
            z2.sum().backward()
        # print(ray_result.d[0, :2, :2, :], ray_result2.d[0, :2, :2, :])
        print("triton time cost:", time_cost.intervalToNow())
        print("z diff:", (z - z2).abs().max())
        print("c grad diff:", clen.c.grad, nclen.c.grad)
        print("ai2 grad diff:", clen.ai2.grad, nclen.ai2.grad)
        print("ai4 grad diff:", clen.ai4.grad, nclen.ai4.grad)
        print("ai6 grad diff:", clen.ai6.grad, nclen.ai6.grad)
        print("ai8 grad diff:", clen.ai8.grad, nclen.ai8.grad)
        print("ai10 grad diff:", clen.ai10.grad, nclen.ai10.grad)
        print("ai12 grad diff:", clen.ai12.grad, nclen.ai12.grad)
        # print(clen.ai2.dtype)
        print("o grad diff:", (tmp_ray1.o.grad-tmp_ray2.o.grad).abs().max())
        print(tmp_ray2.o.grad[...,0][0,:2,...])


def test_ray_reaction_fwd_multi(lens, rays_backup):
    clen = lens.surfaces[1]
    nclen = clen.clone()
    ray_results, ray_results2 = [], []
    print("rays_backupp num:", len(rays_backup))
    # for i, wv in enumerate(WAVE_RGB[:1]):
    ray = rays_backup[0].clone()
    ray.o.requires_grad = True
    ray.d.requires_grad = True
    ray.ra.requires_grad = True
    ray.obliq.requires_grad = True
    tmp1_o, tmp1_d = ray.o, ray.d
    mat1 = lens.surfaces[0].mat2
    n1 = mat1.ior(ray.wvln)
    n2 = clen.mat2.ior(ray.wvln)
    # ray = clen.intersect(ray, n1, method="old")
    # ray = clen.refract(ray, n1 / n2)
    ray = clen.ray_reaction(ray, n1, n2)
    ray_results.append(ray)

    # for i, wv in enumerate(WAVE_RGB[:1]):
    #     ray = rays_backup[i].clone()
    #     mat1 = lens.surfaces[0].mat2
    #     n1 = mat1.ior(ray.wvln)
    #     n2 = nclen.mat2.ior(ray.wvln)
    #     # ray = nclen.intersect(ray, n1, method="new")
    #     ray = nclen.refract_new(ray, n1 / n2)
    #     # ray = nclen.ray_reaction(ray, n1, n2, method="new", refactor="new")
    #     ray_results2.append(ray)

    merged_results = merge_rays(ray_results)
    # merged_results.o.sum() ++ merged_results.ra.sum() + merged_results.obliq.sum()
    (ray.d.sum() + ray.o.sum() + ray.ra.sum() + ray.obliq.sum()).backward()
    # merged_results2 = merge_rays(ray_results2)

    ray2 = rays_backup[0].clone()
    ray2.o.requires_grad = True
    ray2.d.requires_grad = True
    ray2.ra.requires_grad = True
    ray2.obliq.requires_grad = True
    # tmp2_o, tmp2_d = ray2.o, ray2.d
    # rays_backup_clone = [ray.clone() for ray in rays_backup[:1]]
    rays_backup_clone = [ray2]

    merged_rays = merge_rays(rays_backup_clone)
    tmp2_o, tmp2_d = merged_rays.o, merged_rays.d
    tmp2_d.retain_grad(), tmp2_o.retain_grad()
    merged_results2 = nclen.ray_reaction_multi_waves(merged_rays)
    # merged_results2.o.sum() + +merged_results2.ra.sum() + merged_results2.obliq.sum()
    (merged_results2.d.sum() + merged_results2.o.sum() + merged_results2.ra.sum() + merged_results2.obliq.sum()).backward()
    print("d diff:", (merged_results.d - merged_results2.d).abs().max())
    # print("d", merged_results.d[0, 0, 0, :2, :], merged_results2.d[0, 0, 0, :2, :])
    print("o diff:", (merged_results.o - merged_results2.o).abs().max())
    print("ra diff:", (merged_results.ra - merged_results2.ra).abs().max(), merged_results2.ra.min())
    print("obliq diff:", (merged_results.obliq - merged_results2.obliq).abs().max())
    # print("obliq", merged_results.obliq[0, 0, :2, :2], merged_results2.obliq[0, 0, :2, :2])
    print("o grad diff:", (tmp1_o.grad - tmp2_o.grad.squeeze(0)).abs().max())
    print("d grad diff:", (tmp1_d.grad - tmp2_d.grad.squeeze(0)).abs().max())
    print("d grad", tmp1_d.grad[0, 0, :2, :], tmp2_d.grad.squeeze(0)[0, 0, :2, :])
    print("c grad diff:", clen.c.grad, nclen.c.grad)
    print("d grad diff:", clen.d.grad, nclen.d.grad)
    print("ai2 grad diff:", clen.ai2.grad, nclen.ai2.grad)
    print("ai4 grad diff:", clen.ai4.grad, nclen.ai4.grad)
    print("ai6 grad diff:", clen.ai6.grad, nclen.ai6.grad)
    print("ai8 grad diff:", clen.ai8.grad, nclen.ai8.grad)
    print("ai10 grad diff:", clen.ai10.grad, nclen.ai10.grad)
    print("ai12 grad diff:", clen.ai12.grad, nclen.ai12.grad)


def merge_rays(rays_backup):
    o = torch.stack([ray.o for ray in rays_backup])
    d = torch.stack([ray.d for ray in rays_backup])
    ra = torch.stack([ray.ra for ray in rays_backup])
    obliq = torch.stack([ray.obliq for ray in rays_backup])
    print(o.shape, d.shape)
    merged_ray = Ray(o, d, device=o.device)
    merged_ray.ra = ra
    merged_ray.obliq = obliq
    return merged_ray


def unitTest(clen, ray):
    tmp_ray1 = ray.clone()
    tmp_ray1.o.requires_grad = True
    tmp_ray1.d.requires_grad = True
    tmp_ray1.ra.requires_grad = True
    tmp_ray2 = ray.clone()
    tmp_ray2.o.requires_grad = True
    tmp_ray2.d.requires_grad = True
    tmp_ray2.ra.requires_grad = True
    nclen = clen.clone()
    t, valid = clen.newtons_method_test(tmp_ray1)
    t.sum().backward()
    d_grad, c_grad, ai2_grad, ai4_grad, ai6_grad, ai8_grad, ai10_grad, ai12_grad = (
        clen.d.grad,
        clen.c.grad,
        clen.ai2.grad,
        clen.ai4.grad,
        clen.ai6.grad,
        clen.ai8.grad,
        clen.ai10.grad,
        clen.ai12.grad,
    )

    t_2, valid_2 = nclen.newtons_method2(tmp_ray2)
    t_2.sum().backward()
    nd_grad, nc_grad, nai2_grad, nai4_grad, nai6_grad, nai8_grad, nai10_grad, nai12_grad = (
        nclen.d.grad,
        nclen.c.grad,
        nclen.ai2.grad,
        nclen.ai4.grad,
        nclen.ai6.grad,
        nclen.ai8.grad,
        nclen.ai10.grad,
        nclen.ai12.grad,
    )
    print(
        "==== diff",
        torch.max(torch.abs(t - t_2.squeeze(-1))),
        torch.max(torch.abs(valid.to(torch.int32) - valid_2.squeeze(-1).to(torch.int32))),
        torch.max(torch.abs(tmp_ray1.o.grad - tmp_ray2.o.grad)),
        torch.max(torch.abs(tmp_ray1.d.grad - tmp_ray2.d.grad)),
        d_grad - nd_grad,
        c_grad - nc_grad,
        ai2_grad - nai2_grad,
        ai4_grad - nai4_grad,
        ai6_grad - nai6_grad,
        ai8_grad - nai8_grad,
        ai10_grad - nai10_grad,
        ai12_grad - nai12_grad,
    )


def zero_grad(ray, clen):
    ray.o.grad.zero_()
    ray.d.grad.zero_()
    clen.d.grad.zero_()
    clen.c.grad.zero_()
    clen.ai2.grad.zero_()
    clen.ai4.grad.zero_()
    clen.ai6.grad.zero_()
    clen.ai8.grad.zero_()
    clen.ai10.grad.zero_()
    clen.ai12.grad.zero_()


def test_surface_0(lens, ray):
    mat1 = Material("air")
    n1 = mat1.ior(ray.wvln)
    n2 = lens.surfaces[0].mat2.ior(ray.wvln)
    print(ray.o.shape, ray.o.device)
    time_cost = TimeMeter()
    for j in range(3):
        n1 = mat1.ior(ray.wvln)
        n2 = lens.surfaces[0].mat2.ior(ray.wvln)
        time_cost.start()
        for i in range(1000):
            tmp_ray1 = ray.clone()
            rray = lens.surfaces[0].ray_reaction(tmp_ray1, n1, n2)

        print("0 time cost:", time_cost.intervalToNow())
        time_cost.reset()
        time_cost.start()
        for i in range(1000):
            tmp_ray2 = ray.clone()
            lens.surfaces[0].ray_reaction(tmp_ray2, n1, n2)
        print("max diff", torch.max(torch.abs(rray.o - tmp_ray2.o)))
        print("triton time cost:", time_cost.intervalToNow())
        # mat1 = lens.surfaces[1].mat2
        # n1 = mat1.ior(ray.wvln)
        # n2 = lens.surfaces[1].mat2.ior(ray.wvln)
        # time_cost.start()
        # for i in range(1):
        #     tmp_ray = t.clone()
        #     t = lens.surfaces[1].ray_reaction(tmp_ray, n1, n2)
        # print("1 time cost:", time_cost.intervalToNow())
        time_cost.reset()


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(15)
    # lens, ray, rays_backup = init_from_file()
    lens, ray, rays_backup = init()

    # test_surface_1_bwd(lens, ray)
    # test_surface_1_fwd(lens, ray)
    # test_ray_reaction_bwd(lens, ray)
    #### ray refraction正确
    # test_ray_reaction_fwd_multi(lens, rays_backup)
    # test_ray_intersection_bwd(lens, ray)
    test_surface_sag(lens, ray)
    # test_ray_refraction(lens, ray)
    # test_ray_refraction_bwd(lens, ray)


if __name__ == "__main__":
    main()
