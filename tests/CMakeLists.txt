foreach(dir IN LISTS TASK_INCLUDE_DIRS)
    include_directories(${dir})
endforeach()

include_directories(${RNG_INCLUDE_DIRS})

file(GLOB tests RELATIVE 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}/*
)

message(STATUS "[#] TESTS")

foreach(test IN LISTS tests)
    set(testPath ${CMAKE_CURRENT_SOURCE_DIR}/${test})
    if (IS_DIRECTORY ${testPath})
        if (EXISTS ${testPath}/main.cpp)
            file(GLOB SOURCE_FILES RELATIVE
                ${CMAKE_CURRENT_SOURCE_DIR}/${test}
                ${CMAKE_CURRENT_SOURCE_DIR}/${test}/*
            )

            set(SOURCE_FILES_PATHS)
            foreach(file IN LISTS SOURCE_FILES)
                string(PREPEND file "${test}/")
                list(APPEND SOURCE_FILES_PATHS ${file})
            endforeach()

            add_executable(test_${test} ${SOURCE_FILES_PATHS})
            set_target_properties(test_${test} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test
            )
            target_link_libraries(test_${test}
                PRIVATE gtest
                PRIVATE gtest_main
                PRIVATE rng
                PRIVATE ${test}
            )

            message(STATUS "- test_${test}")
        endif()
    endif()
endforeach()
