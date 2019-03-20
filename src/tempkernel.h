#ifndef __TEMPKERNEL_H_
#define __TEMPKERNEL_H_

//this macro avoids commas in expanding functions
#define TEMPLATE_QUEUE_NORMAL(func) func<Queue,Normal>
#define TEMPLATE_QUEUE_CACHED(func) func<Queue,Cached>
#define TEMPLATE_BITMAP_NORMAL(func) func<Bitmap,Normal>

// this macro avoids commas in inspecting functions
#define TEMPLATE_G_F_CSR(inspect_func) inspect_func<device_graph_t<CSR,E>,F>
#define TEMPLATE_G_F_COO(inspect_func) inspect_func<device_graph_t<COO,E>,F>

// this macro avoids commas in sorted search TODO: may be removed!
#define TEMPLATE_TSZ(func) func<data_t, TILE_SZ>

#endif