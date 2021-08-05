#!/usr/bin/env python3
""" creates class normal """


class Normal:
    """ creates class that represents a normal distribution """
    def __init__(self, data=None, mean=0., stddev=1.):
        """ constructor class
        data is a list of the data to be used to estimate the distribution
        mean is the mean of the distribution
        stddev is the standard deviation of the distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = float(sum(data) / len(data))
                summation = 0
                for i in data:
                    summation += pow(i - self.mean, 2)
                self.stddev = pow(summation / len(data), 1/2)

    def z_score(self, x):
        """"
        Instance method which calculates the z-score of a given x-value
        x is the x-value
        Returns the z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Instance method which calculates the x-value of a given z-score
        z is the z-score
        Returns the x-value of z
        """
        return self.mean + (self.stddev * z)

    def pdf(self, x):
        """
        Instance method which calculates the value of the PDF
        for a given x-value
        x is the x-value
        Returns the PDF value for x
        """
        pi = 3.1415926536
        e = 2.7182818285
        sd = self.stddev
        variance = pow(sd, 2)
        mi = self.mean
        exp = -pow(x - mi, 2) / (2 * variance)
        denom = sd * pow(2 * pi, 1/2)
        pdf_normal = pow(e, exp) / denom

        return pdf_normal
    
    def cdf(self, x):
        """
        Instance method which calculates the Cumulative Distribution
        Function for an x-given value
        Calculated using the error function which is calculated as a 
        Maclaurin series
        Args:
            x ([int]): x-value for which the CDF will be calculated
        Return: CDF for x value
        """
        pi = 3.1415926536
        z = (x - self.mean) / (self.stddev * pow(2, 1 / 2))
        erf = pow(z, 3) / 3 + pow(z, 5) / 10 - pow(z, 7) / 42 + pow(z, 9) / 216 
        cdf = (1 + 2 / pow(pi, 1/2) * erf) / 2
        return cdf
