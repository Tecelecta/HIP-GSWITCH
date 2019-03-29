#ifndef __TEMPKERNEL_H_
#define __TEMPKERNEL_H_

//this macro avoids commas in expanding functions
#define TSPEC_QUEUE_NORMAL(func) func<Queue,Normal>
#define TSPEC_QUEUE_CACHED(func) func<Queue,Cached>
#define TSPEC_BITMAP_NORMAL(func) func<Bitmap,Normal>

// this macro avoids commas in inspecting functions
#define TSPEC_G_F_CSR(inspect_func) inspect_func<device_graph_t<CSR,E>,F>
#define TSPEC_G_F_COO(inspect_func) inspect_func<device_graph_t<COO,E>,F>

#endif