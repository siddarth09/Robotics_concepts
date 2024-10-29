cmake_minimum_required(VERSION 3.8)
project(gtsam_example)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(GTSAM REQUIRED)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS}
                    ${GTSAM_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS})

add_executable(tut1 src/tut1.cpp)
ament_target_dependencies(tut1 Eigen3 GTSAM)
target_link_libraries(tut1
gtsam 
gtsam_unstable
)

add_executable(localization src/localization.cpp)
ament_target_dependencies(localization Eigen3 GTSAM)
target_link_libraries(localization
gtsam 
gtsam_unstable
)


install(TARGETS
  tut1
  localization
  DESTINATION lib/${PROJECT_NAME})


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
