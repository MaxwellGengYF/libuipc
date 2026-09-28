if(UIPC_WITH_CUDA_BACKEND)
    add_subdirectory(cuda)
endif()

if(UIPC_WITH_LUISA_BACKEND)
    add_subdirectory(luisa)
endif()
