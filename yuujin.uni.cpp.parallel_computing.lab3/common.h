#pragma once

#include <iostream>

double function(double x)
{
	return log(2.0 + sin(10.0 * x)) / sqrt(x + 1.0);
}