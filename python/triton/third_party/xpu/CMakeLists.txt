
# Python module
if(TRITON_BUILD_PYTHON_MODULE)
    message(STATUS "Adding Intel XPU Backend Python module")
    set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
    set(PYTHON_SRC ${PYTHON_SRC_PATH}/main.cc PARENT_SCOPE)

    file(INSTALL
            ${CMAKE_CURRENT_SOURCE_DIR}/lib/libsycl-spir64-unknown-unknown.bc
            DESTINATION ${PYTHON_THIRD_PARTY_PATH}/xpu/lib)

    file(INSTALL
            ${CMAKE_CURRENT_SOURCE_DIR}/utils/sycl.cpp
            DESTINATION ${PYTHON_THIRD_PARTY_PATH}/xpu/utils)

    file(INSTALL
            ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
            ${CMAKE_CURRENT_SOURCE_DIR}/extensions.py
            DESTINATION ${PYTHON_THIRD_PARTY_PATH}/xpu/)
endif()
