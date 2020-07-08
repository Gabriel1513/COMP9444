def myPow(x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n == 0:
            return 1
        if n < 0:
            return 1.0/myPow(x,-n)
        a = n//2
        half = myPow(x,a)
        if n%2 == 0:
            return half*half
        else:
            return half*half*x
