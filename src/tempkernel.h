#ifndef __TEMPKERNEL_H_
#define __TEMPKERNEL_H_

#define TEMPLATE_QUEUE_NORMAL(func) func<Queue,Normal>
#define TEMPLATE_QUEUE_CACHED(func) func<Queue,Cached>

#define TEMPLATE_BITMAP_NORMAL(func) func<Bitmap,Normal>


#endif