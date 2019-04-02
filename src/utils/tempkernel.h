#ifndef __TEMPKERNEL_H_
#define __TEMPKERNEL_H_

//this macro avoids commas in expanding functions
#define TSPEC_QUEUE_NORMAL(func) func<Queue,Normal>
#define TSPEC_QUEUE_CACHED(func) func<Queue,Cached>
#define TSPEC_BITMAP_NORMAL(func) func<Bitmap,Normal>

#endif