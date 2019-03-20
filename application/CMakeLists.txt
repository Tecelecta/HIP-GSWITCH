HIP_ADD_EXECUTABLE(BFS bfs.cpp)
target_link_libraries(BFS gswitch)

HIP_ADD_EXECUTABLE(CC cc.cpp)
target_link_libraries(CC gswitch)

HIP_ADD_EXECUTABLE(PageRank pagerank.cpp)
target_link_libraries(PageRank gswitch)

HIP_ADD_EXECUTABLE(BC bc.cpp)
target_link_libraries(BC gswitch)

HIP_ADD_EXECUTABLE(SSSP sssp.cpp)
target_link_libraries(SSSP gswitch)

HIP_ADD_EXECUTABLE(GC gc.cpp)
target_link_libraries(GC gswitch)

#HIP_ADD_EXECUTABLE(BF bf.cpp)
#target_link_libraries(BF gswitch)

#HIP_ADD_EXECUTABLE(Delta-PageRank delta-pagerank.cpp)
#target_link_libraries(Delta-PageRank gswitch)
#
HIP_ADD_EXECUTABLE(Test test.cpp)
target_link_libraries(Test gswitch)
