#include "regions.h"

rectangle::rectangle(long x_start, long y_start, long x_size, long y_size)  {
	rectangle::x_start = x_start;
	rectangle::y_start = y_start;
	rectangle::x_size = x_size;
	rectangle::y_size = y_size;
};

rectangle rectangle::operator-(const rectangle& r) {
	rectangle new_rectangle;
	new_rectangle.x_start = rectangle::x_start - r.x_start;
	new_rectangle.y_start = rectangle::y_start - r.y_start;
	new_rectangle.x_size = rectangle::x_size;
	new_rectangle.y_size = rectangle::y_size;
	return new_rectangle;
}