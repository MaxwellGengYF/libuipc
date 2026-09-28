if has_config("backend_cuda") then
    includes("cuda")
end

if has_config("backend_luisa") then
    includes("luisa")
end
