/*
 * The MIT License
 *
 * Copyright 2019 Rob Garcia at rgarcia@rgprogramming.com.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package mlalgorithmsinjava;

/**
 *
 * @author Rob Garcia at rgarcia@rgprogramming.com
 */
public class LinearRegression {
    /** USE DOUBLE WRAPPER, NOT PRIMITIVE TYPE! WE NEED TO DO NULL CHECKS! **/
    
    /**
     * Calculate a hypothesis for a linear regression problem.
     * @param theta a vector of weights. theta[0] is the y-intercept (b) and theta[1...n] are the slopes of each x value
     * @param x a vector of x values from the X matrix
     * @return the hypothesis, i.e., the estimated y value
     */
    public static Double Hypothesis(Double[] theta, Double[] x) {
        if(theta.length != x.length) {
            System.out.println("Check your data: the number of x elements (" + x.length + ") does not equal the number of elements in theta transpose (" + theta.length + ").");
            return null;
        }
        else {
            Double h = 0.0;
            for(int i = 0; i < x.length; i++) {
                h += theta[i] * x[i];
            }
            return h;
        }
    }
    
    /**
     * Calculate the cost for a linear regression problem. A result of 0 is the goal.
     * @param h a vector of results of multiple hypothesis calculations
     * @param y the values that h is trying to predict
     * @return J the cost or difference between the predicted values and the actual values
     */
    public static Double CostFunction(Double[] h, Double[] y) {
        if(h.length != y.length) {
            System.out.println("Check your data: The number of h's (" + h.length + ") does not equal the number of y's (" + y.length + ").");
            return null;
        }
        else {
            Double sumOfSquares = 0.0;
            for(int i = 0; i < h.length; i++) {
                sumOfSquares += Math.pow((h[i] - y[i]), 2);
            }
            Double m = (double)h.length;
            Double J = (1.0 / (2.0 * m)) * sumOfSquares;
            System.out.printf("J = %.2f = 1/(2 * %.0f) * \u221A(%.2f)\u00B2 \n", J, m, sumOfSquares);
            return J;
        }
    }
    
    public static Double GradientDescent(Double[] theta) {

        return null;
    }
}
