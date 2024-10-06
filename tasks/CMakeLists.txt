file(GLOB tasks RELATIVE 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}/*
)

include_directories(${RNG_INCLUDE_DIRS})

message(STATUS "[#] EXECUTABLES/LIBS")

foreach(task IN LISTS tasks)
    set(taskPath ${CMAKE_CURRENT_SOURCE_DIR}/${task})
    if (IS_DIRECTORY ${taskPath})
        list(APPEND TASK_INCLUDE_DIRS ${taskPath})
        string(REGEX REPLACE "^[0-9]+_" "" taskName ${task})
        if (EXISTS ${taskPath}/${task}.cpp)
            if (EXISTS ${taskPath}/${taskName}.cpp)
                set(sourceFile ${taskName}.cpp)
            elseif(EXISTS ${taskPath}/${taskName}.cu)
                set(sourceFile ${taskName}.cu)
            endif()
            add_executable(app_${task} ${taskPath}/${task}.cpp)
            add_library(${task} STATIC ${taskPath}/${sourceFile})
            target_link_libraries(app_${task} PRIVATE 
                ${task}
                rng
            )
            if(${taskName} MATCHES ".*omp.*")
                target_link_libraries(${task} PRIVATE OpenMP::OpenMP_CXX)
                # SIMD 
                target_compile_options(${task} PRIVATE -march=native)
            endif()
            if(${taskName} MATCHES ".*cuda.*")
                target_compile_options(${task} PRIVATE "-G")
            endif()

            message(STATUS "- app_${task} / ${task}")
        endif()
    endif()
endforeach()

set(TASK_INCLUDE_DIRS ${TASK_INCLUDE_DIRS} PARENT_SCOPE)
