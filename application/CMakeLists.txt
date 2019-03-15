HIP_ADD_EXECUTABLE(BFS bfs.cu)
target_link_libraries(BFS gswitch)

HIP_ADD_EXECUTABLE(CC cc.cu)
target_link_libraries(CC gswitch)

HIP_ADD_EXECUTABLE(PageRank pagerank.cu)
target_link_libraries(PageRank gswitch)

HIP_ADD_EXECUTABLE(BC bc.cu)
target_link_libraries(BC gswitch)

HIP_ADD_EXECUTABLE(SSSP sssp.cu)
target_link_libraries(SSSP gswitch)

HIP_ADD_EXECUTABLE(GC gc.cu)
target_link_libraries(GC gswitch)

#HIP_ADD_EXECUTABLE(BF bf.cu)
#target_link_libraries(BF gswitch)

#HIP_ADD_EXECUTABLE(Delta-PageRank delta-pagerank.cu)
#target_link_libraries(Delta-PageRank gswitch)
#
HIP_ADD_EXECUTABLE(Test test.cu)
target_link_libraries(Test gswitch)
