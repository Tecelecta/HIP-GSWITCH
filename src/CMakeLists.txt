

set(SROUCE 
  utils/2d_partition.hxx
  utils/cmdline.hxx
  utils/common.hxx
  utils/intrinsics.hxx
  utils/utils.hxx
  utils/block_scan.hxx
  utils/scan.hxx
  utils/fileIO.hxx
  utils/json.hxx
  data_structures/active_set.hxx
  data_structures/bitmap.hxx
  data_structures/queue.hxx
  data_structures/window.hxx
  data_structures/workset.hxx
  data_structures/functor.hxx
  data_structures/graph.hxx
  data_structures/notification.hxx
  kernel_libs/filter.hxx
  kernel_libs/expand_EC.hxx
  kernel_libs/expand_VC_CM.hxx
  kernel_libs/expand_VC_STRICT.hxx
  kernel_libs/expand_VC_TM.hxx
  kernel_libs/expand_VC_TWC.hxx
  kernel_libs/expand_VC_TWOD.hxx
  kernel_libs/expand_VC_WM.hxx
  kernel_libs/kernel_fusion.hxx
  abstraction/config.hxx
  abstraction/statistics.hxx
  abstraction/features.hxx
  abstraction/inspector.hxx
  abstraction/selector.hxx
  abstraction/executor.hxx
  gswitch.h)


#HIP_ADD_EXECUTABLE(gswitch ${SROUCE})
HIP_ADD_LIBRARY(gswitch ${SROUCE} STATIC)
