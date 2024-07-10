#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Branch and bound for input space split."""

import time
import torch
import math
import sys

import arguments
from auto_LiRPA.utils import (stop_criterion_batch_any, stop_criterion_all,
                              AutoBatchSize)
from utils import check_auto_enlarge_batch_size
from input_split.branching_domains import UnsortedInputDomainList
from input_split.attack import (massive_pgd_attack, check_adv,
                                attack_in_input_bab_parallel,
                                update_rhs_with_attack)
from input_split.branching_heuristics import input_split_branching, clip_domains
from input_split.split import input_split_parallel, get_split_depth
from input_split.utils import transpose_c_back, initial_verify_criterion

from collections import defaultdict

Visited, storage_depth = 0, 0


def batch_verification_input_split(
        d, net, batch, num_iter, decision_thresh, shape=None,
        bounding_method="crown", branching_method="sb",
        stop_func=stop_criterion_batch_any, split_partitions=2):
    input_split_args = arguments.Config["bab"]["branching"]["input_split"]
    split_hint = input_split_args["split_hint"]

    split_start_time = time.time()
    global Visited

    # STEP 1: pick out domains
    pickout_start_time = time.time()
    ret = d.pick_out_batch(batch, device=net.x.device)
    alphas, dm_lb, x_L, x_U, cs, thresholds, split_idx = ret
    pickout_time = time.time() - pickout_start_time

    if input_split_args["update_rhs_with_attack"]:
        thresholds = update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb,
                                            net.model_ori)

    # STEP 2: create new split domains.
    decision_start_time = time.time()
    split_depth = get_split_depth(x_L, split_partitions=split_partitions)
    new_x_L, new_x_U, cs, thresholds, split_depth = input_split_parallel(
        x_L, x_U, shape, cs, thresholds, split_depth=split_depth, i_idx=split_idx,
        split_partitions=split_partitions, split_hint=split_hint)

    if input_split_args["compare_with_old_bounds"]:
        assert split_depth == 1
        dm_lb = dm_lb.repeat(2, *[1]*(dm_lb.ndim - 1))
    else:
        dm_lb = None

    alphas = alphas * (split_partitions ** (split_depth - 1))

    decision_time = time.time() - decision_start_time

    # STEP 3: Compute bounds for all domains and make decisions.
    new_x_L, new_x_U, new_dm_lb, alphas, split_idx, bounding_time, decision_time_, clip_time = get_bound_and_decision(
        net, dm_lb, new_x_L, new_x_U, alphas, cs, thresholds,
        bounding_method, branching_method, stop_func, num_iter
    )
    decision_time += decision_time_

    # STEP 4: Add new domains back to domain list.
    adddomain_start_time = time.time()
    d.add(new_dm_lb, new_x_L.detach(), new_x_U.detach(),
          alphas, cs, thresholds, split_idx)
    adddomain_time = time.time() - adddomain_start_time

    total_time = time.time() - split_start_time
    print(
        f"Total time: {total_time:.4f}  pickout: {pickout_time:.4f}  "
        f"decision: {decision_time:.4f}  bounding: {bounding_time:.4f}  "
        f"clipping: {clip_time:.4f}  add_domain: {adddomain_time:.4f}"
    )
    print("Length of domains:", len(d))

    Visited += len(new_x_L)
    print(f"{Visited} branch and bound domains visited")

    if len(d) == 0:
        print("No domains left, verification finished!")
        if new_dm_lb is not None:
            print(f"The lower bound of last batch is {new_dm_lb.min().item()}")
        return decision_thresh.max() + 1e-7
    else:
        if input_split_args["skip_getting_worst_domain"]:
            # It can be costly to call get_topk_indices when the domain list is long
            worst_idx = 0
        else:
            worst_idx = d.get_topk_indices().item()
        worst_val = d[worst_idx]
        global_lb = worst_val[0] - worst_val[-1]
        if not input_split_args["skip_getting_worst_domain"]:
            if 1 < global_lb.numel() <= 5:
                print(f"Current (lb-rhs): {global_lb}")
            else:
                print(f"Current (lb-rhs): {global_lb.max().item()}")

    if input_split_args["show_progress"]:
        print(f"Progress: {d.get_progess():.10f}")
    sys.stdout.flush()

    return global_lb


def get_bound_and_decision(net, dm_lb, x_L, x_U, alphas, cs, thresholds,
                           bounding_method, branching_method, stop_func,
                           num_iter):
    bounding_start_time = time.time()
    ret = net.get_lower_bound_naive(
        dm_lb=dm_lb, dm_l=x_L, dm_u=x_U, alphas=alphas,
        bounding_method=bounding_method, branching_method=branching_method,
        C=cs, stop_criterion_func=stop_func, thresholds=thresholds)
    new_dm_lb, alphas, lA, lbias = ret  # here alphas is a dict
    bounding_time = time.time() - bounding_start_time

    new_dm_lb = new_dm_lb.to(device=thresholds.device)  # ensures it is on the same device as it may be different

    # shrink these new domains
    enable_clip_domains = arguments.Config["bab"]["branching"]["input_split"]["enable_clip_domains"]
    clip_time = 0.
    if enable_clip_domains:
        clip_start_time = time.time()
        ret = clip_domains(x_L, x_U, thresholds, lA, new_dm_lb)
        x_L, x_U = ret
        clip_time = time.time() - clip_start_time

    decision_start_time = time.time()
    split_idx = input_split_branching(
        net, new_dm_lb, x_L, x_U, lA, thresholds,
        branching_method, storage_depth, num_iter=num_iter
    )
    decision_time = time.time() - decision_start_time

    return x_L, x_U, new_dm_lb, alphas, split_idx, bounding_time, decision_time, clip_time


def input_bab_parallel(net, init_domain, x, rhs=None,
                       timeout=None, max_iterations=None,
                       vnnlib=None, c_transposed=False, return_domains=False):
    """Run input split bab.

    c_transposed: bool, by default False, indicating whether net.c matrix has
        transposed between dim=0 and dim=1. As documented in abcrown.py bab(),
        if using input split, and if there are multiple specs with shared single input,
        we transposed the c matrix from [multi-spec, 1, ...] to [1, multi-spec, ...] so that
        net.build() process in input_bab_parallel could share intermediate layer
        bounds across all specs. If such transpose happens, c_transposed is set
        to True, so that after net.build() in this func, we can transpose c
        matrix back, repeat x_LB & x_UB, duplicate alphas, to prepare for input domain bab.
    """
    global storage_depth

    start = time.time()
    # All supported arguments.
    global Visited

    bab_args = arguments.Config["bab"]
    branching_args = bab_args["branching"]
    input_split_args = branching_args["input_split"]

    timeout = timeout or bab_args["timeout"]
    batch = arguments.Config["solver"]["batch_size"]
    bounding_method = arguments.Config["solver"]["bound_prop_method"]
    init_bounding_method = arguments.Config["solver"]["init_bound_prop_method"]
    max_iterations = max_iterations or bab_args["max_iterations"]
    sort_domain_iter = bab_args["sort_domain_interval"]
    branching_method = branching_args["method"]
    adv_check = input_split_args["adv_check"]
    split_partitions = input_split_args["split_partitions"]
    catch_assertion = input_split_args["catch_assertion"]
    use_clip_domains = input_split_args["enable_clip_domains"]

    if init_bounding_method == "same":
        init_bounding_method = bounding_method

    if c_transposed or net.c.shape[1] == 1:
        # When c_transposed applied, previous checks have ensured that there is only single spec,
        # but we compressed multiple data samples to the spec dimension by transposing C
        # so we need "all" to be satisfied to stop.
        # Another case is there is only single spec, so batch_any equals to all. Then, we set to all
        # so we can allow prune_after_crown optimization
        stop_func = stop_criterion_all
    else:
        # Possibly multiple specs in each data sample
        stop_func = stop_criterion_batch_any

    Visited = 0

    dm_l = x.ptb.x_L
    dm_u = x.ptb.x_U

    if (dm_u - dm_l > 0).int().sum() == 1:
        branching_method = "naive"

    try:
        global_lb, ret = net.build(
            init_domain, x, stop_criterion_func=stop_func(rhs),
            bounding_method=init_bounding_method, decision_thresh=rhs, return_A=False)
    except AssertionError:
        if catch_assertion:
            global_lb = torch.ones(net.c.shape[0], net.c.shape[1],
                                   device=net.device) * torch.inf
            ret = {"alphas": {}}
        else:
            raise

    if getattr(net.net[net.net.input_name[0]], "lA", None) is not None:
        lA = net.net[net.net.input_name[0]].lA.transpose(0, 1)
    else:
        lA = None
        if bounding_method == "sb":
            raise ValueError("sb heuristic cannot be used without lA.")

    if c_transposed:
        lA, global_lb, rhs, dm_l, dm_u = transpose_c_back(
            lA, global_lb, rhs, dm_l, dm_u, ret, net)

    result = "unknown"

    # shrink the initial dm_l and dm_u
    if use_clip_domains:
        dm_l, dm_u = clip_domains(dm_l, dm_u, rhs, lA, global_lb)

    # compute storage depth
    use_alpha = init_bounding_method.lower() == "alpha-crown" or bounding_method == "alpha-crown"
    min_batch_size = (
        arguments.Config["solver"]["min_batch_size_ratio"]
        * arguments.Config["solver"]["batch_size"]
    )
    max_depth = max(int(math.log(max(min_batch_size, 1)) // math.log(split_partitions)), 1)
    storage_depth = min(max_depth, dm_l.shape[-1])
    domains = UnsortedInputDomainList(
        storage_depth, use_alpha=use_alpha,
        sort_index=input_split_args["sort_index"],
        sort_descending=input_split_args["sort_descending"])

    initial_verified, remaining_index = initial_verify_criterion(global_lb, rhs)
    if initial_verified:
        result = "safe"
    else:
        # compute initial split idx
        split_idx = input_split_branching(
            net, global_lb, dm_l, dm_u, lA, rhs, branching_method, storage_depth)
        domains.add(global_lb, dm_l.detach(), dm_u.detach(),
                    ret["alphas"], net.c, rhs, split_idx, remaining_index)
        if arguments.Config["attack"]["pgd_order"] == "after":
            if attack_in_input_bab_parallel(net.model_ori, domains, x, vnnlib=vnnlib):
                print("pgd attack succeed in input_bab_parallel")
                result = "unsafe"
        if input_split_args["presplit_domains"]:
            assert not use_alpha
            load_presplit_domains(
                domains, net, bounding_method, branching_method, stop_func,
            )

    if arguments.Config["solver"]["auto_enlarge_batch_size"]:
        auto_batch_size = AutoBatchSize(batch, net.device)
    else:
        auto_batch_size = None

    num_iter = 1
    enhanced_bound_initialized = False
    while (result == "unknown" and len(domains) > 0
           and (max_iterations == -1 or num_iter <= max_iterations)):
        print(f"Iteration {num_iter}")
        # sort the domains every certain number of iterations
        if sort_domain_iter > 0 and num_iter % sort_domain_iter == 0:
            sort_start_time = time.time()
            domains.sort()
            sort_time = time.time() - sort_start_time
            print(f"Sorting domains used {sort_time:.4f}s")

        last_glb = global_lb.max()

        if arguments.Config["attack"]["pgd_order"] != "skip":
            if adv_check != -1 and Visited > adv_check:
                adv_check_start_time = time.time()
                # check whether adv example found
                if check_adv(domains, net.model_ori, x, vnnlib=vnnlib):
                    return global_lb.max(), Visited, "unsafe"
                adv_check_time = time.time() - adv_check_start_time
                print(f"Adv attack time: {adv_check_time:.4f}s")

        batch_ = batch
        if branching_method == "brute-force" and num_iter <= input_split_args["bf_iters"]:
            batch_ = input_split_args["bf_batch_size"]
        print("Batch size:", batch_)
        if auto_batch_size:
            auto_batch_size.record_actual_batch_size(min(batch_, len(domains)))
        try:
            global_lb = batch_verification_input_split(
                domains, net, batch_,
                num_iter=num_iter, decision_thresh=rhs, shape=x.shape,
                bounding_method=bounding_method, branching_method=branching_method,
                stop_func=stop_func, split_partitions=split_partitions)
        except AssertionError:
            if catch_assertion:
                global_lb = torch.ones(net.c.shape[0], net.c.shape[1],
                                       device=net.device) * torch.inf
            else:
                raise
        if auto_batch_size:
            batch = check_auto_enlarge_batch_size(auto_batch_size)

        # once the lower bound stop improving we change to solve alpha mode
        if (arguments.Config["solver"]["bound_prop_method"]
            != input_split_args["enhanced_bound_prop_method"]
            and time.time() - start > input_split_args["enhanced_bound_patience"]
            and global_lb.max().cpu() <= last_glb.cpu()
            and bounding_method != "alpha-crown"
            and not enhanced_bound_initialized
        ):
            enhanced_bound_initialized = True
            global_lb, domains, branching_method, bounding_method = enhanced_bound_init(
                net, init_domain, x, stop_func, rhs, storage_depth, num_iter)

        if arguments.Config["attack"]["pgd_order"] != "skip":
            if time.time() - start > input_split_args["attack_patience"]:
                print("Perform PGD attack with massively random starts finally.")
                ret_adv = massive_pgd_attack(x, net.model_ori, vnnlib=vnnlib)[1]
                if ret_adv:
                    result = "unsafe"
                    break

        if time.time() - start > timeout:
            print("Time out!")
            break

        print(f"Cumulative time: {time.time() - start}\n")
        num_iter += 1

    if result == "unknown" and len(domains) == 0:
        result = "safe"

    if return_domains:
        # Thresholds may have been updated by PGD attack so that different
        # domains may have different thresholds. Restore thresholds to the
        # default RHS for the sorting.
        domains.threshold._storage.data[:] = rhs
        domains.sort()
        if return_domains == -1:
            return_domains = len(domains)
        lower_bound, x_L, x_U = domains.pick_out_batch(
            return_domains, device="cpu")[1:4]
        return lower_bound, x_L, x_U
    else:
        del domains
        return global_lb.max(), Visited, result


def load_presplit_domains(domains, net, bounding_method, branching_method, stop_func):
    input_split_args = arguments.Config["bab"]["branching"]["input_split"]
    batch_size = arguments.Config["solver"]["batch_size"]
    ret = domains.pick_out_batch(len(domains))
    alphas, dm_lb, x_L, x_U, cs, thresholds, split_idx = ret

    presplit_dm_l, presplit_dm_u = torch.load(
        input_split_args["presplit_domains"])
    presplit_dm_l = presplit_dm_l.to(dm_lb)
    presplit_dm_u = presplit_dm_u.to(dm_lb)
    num_presplit_domains = presplit_dm_l.shape[0]
    print(f"Loaded {num_presplit_domains} pre-split domains")

    dm_lb = dm_lb.expand(batch_size, -1)
    cs = cs.expand(batch_size, -1, -1)
    thresholds = thresholds.expand(batch_size, -1)
    num_batches = (num_presplit_domains + batch_size - 1) // batch_size

    for i in range(num_batches):
        print(f"Pre-split domains batch {i+1}/{num_batches}:")
        x_L = presplit_dm_l[i*batch_size:(i+1)*batch_size]
        x_U = presplit_dm_u[i*batch_size:(i+1)*batch_size]
        size = x_L.shape[0]
        x_L, x_U, new_dm_lb, alphas, split_idx, _, _, _ = get_bound_and_decision(
            net, dm_lb[:size], x_L, x_U, alphas, cs[:size], thresholds[:size],
            bounding_method, branching_method, stop_func, num_iter=1
        )
        num_domains_pre = len(domains)
        domains.add(new_dm_lb, x_L, x_U, alphas, cs[:size], thresholds[:size], split_idx)
        print(f"{len(domains) - num_domains_pre} domains added, "
              f"{len(domains)} in total")
        print()

    print(f"{len(domains)} pre-split domains added out of {presplit_dm_l.shape[0]}")
    verified_ratio = 1 - len(domains) * 1. / presplit_dm_l.shape[0]
    print(f"Verified ratio: {verified_ratio}")


def enhanced_bound_init(net, init_domain, x, stop_func, rhs, storage_depth,
                        num_iter):
    input_split_args = arguments.Config["bab"]["branching"]["input_split"]
    branching_method = input_split_args["enhanced_branching_method"]
    bounding_method = input_split_args["enhanced_bound_prop_method"]
    print(f"Using enhanced bound propagation method {bounding_method} "
            f"with {branching_method} branching.")

    global_lb, ret = net.build(
        init_domain, x, stop_criterion_func=stop_func(rhs),
        bounding_method=bounding_method)
    if hasattr(net.net[net.net.input_name[0]], "lA"):
        lA = net.net[net.net.input_name[0]].lA.transpose(0, 1)
    else:
        raise ValueError("sb heuristic cannot be used without lA.")
    dm_l = x.ptb.x_L
    dm_u = x.ptb.x_U

    # compute initial split idx for the enhanced method
    split_idx = input_split_branching(
        net, global_lb, dm_l, dm_u, lA, rhs, branching_method,
        storage_depth, num_iter=num_iter)

    use_alpha = input_split_args["enhanced_bound_prop_method"] == "alpha-crown"
    domains = UnsortedInputDomainList(storage_depth, use_alpha=use_alpha)
    # This is the first batch of initial domain(s) after the branching method changed.
    domains.add(
        global_lb, dm_l.detach(), dm_u.detach(), ret["alphas"],
        net.c, rhs, split_idx)
    global_lb = global_lb.max()

    return global_lb, domains, branching_method, bounding_method
