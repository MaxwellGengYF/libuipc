#!/usr/bin/env bash
# Run every Luisa backend tag batch except the known-slow [fem]/[constitution_a]/
# [constitution_b]/[abd_joint_b] set and the [luisa] umbrella tag (which selects all).
cd /c/dev/libuipc/build/windows/x64/release || exit 1
tags="abd_base abd_constitution abd_constraint abd_dynamics abd_joint_a abd_joint_c abd_linear abd_solve algorithms backend bvh bvh_filter bvh_info constitution_c contact_core converter device distance fem_dynamics fem_linear fem_precond filter flagged frame-stream global_geometry half_plane linear mas_preconditioner math matrix reference spmv t43"
pass=0
fail=""
for t in $tags; do
    out=$(timeout 420 ./uipc_test_backend_luisa.exe "[$t]" 2>&1)
    rc=$?
    line=$(printf '%s\n' "$out" | grep -E "All tests passed|test cases:|FAILED" | tail -2 | tr '\n' ' ')
    if [ $rc -eq 0 ]; then
        pass=$((pass + 1))
        printf 'PASS %-22s %s\n' "$t" "$line"
    else
        fail="$fail $t"
        printf 'FAIL %-22s rc=%d %s\n' "$t" $rc "$line"
        printf '%s\n' "$out" | tail -12
    fi
done
echo "==== $pass passed; failed:${fail:- none} ===="
