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
public class MLAlgorithmsInJava {

    /**
     * x represents a house's square footage
     */
    public static double[][] xMatrix = {{0.0, 500.0}, {0.0, 1000.0}, {0.0, 2000.0}, {0.0, 4000.0}};
    
    /**
     * y represents the actual price of the house based on the square footage ($151 per square foot in 2018)
     */
    public static double[] yVector = {75000.0, 151000.0, 302000.0, 604000.0};
    
    /**
     * mActual is the slope of (x, y): 151/1
     */
    public static double mActual = 151.0;
    
    /**
     * bActual is the y-intercept for (x, y): 0
     */
    public static double bActual = 0.0;
    
    public static double[][] thetaMatrix;
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        System.out.println("Hello, world!");
        System.out.println("Calculating the hypotheses...");
        thetaMatrix = new double[][]{{0.0, 151.0}, {0.0, 149.0}, {0.0, 153.0}};
        double[] thetaVector = new double[thetaMatrix[0].length];
        double[] xVector = new double[xMatrix[0].length];
        double[] h = new double[thetaMatrix.length];
        for(int i = 0; i < thetaMatrix.length; i++) {
            for(int j = 0; j < xMatrix.length; j++) {
                for(int k = 0; k < xMatrix[0].length; k++) {
                    xVector[k] = xMatrix[j][k];
                    thetaVector[k] = thetaMatrix[i][k];
                }
                h[i] = Hypothesis(xVector, thetaVector);
                System.out.printf("h(%.1f) = %.1f + (%.1f * %.1f) = %.1f\n", xVector[1], thetaVector[0], thetaVector[1], xVector[1], h[i]);
            }
            System.out.println();
        }
        System.out.println("Calculating the cost...");
        double[] J = new double[xMatrix.length];
        for(int i = 0; i < thetaMatrix.length; i++) {
            for(int j = 0; j < xMatrix.length; j++) {
                J[i] = CostFunction(h, yVector);
                System.out.printf("J(%.1f, %.1f) = %.1f\n", thetaMatrix[i][0], thetaMatrix[i][1], J[i]);
            }
        }
    }
    
    /**
     * Calculate a hypothesis
     * @param theta a vector of weights. theta[0] is the y-intercept (b) and theta[1...n] are the slopes of each x value
     * @param x a vector of x values from the X matrix
     * @return the hypothesis, i.e., the estimated y value
     */
    public static double Hypothesis(double[] theta, double[] x) {
        double h = 0.0;
        if(theta.length != x.length) {
            System.out.println("x (" + x.length + ") does not equal theta (" + theta.length + ").");
        }
        else {
            for(int i = 0; i < x.length; i++) {
                h += theta[i] * x[i];
            }
        }
        return h;
    }
    
    public static double CostFunction(double[] h, double[] y) {
        double sumOfSquares = 0.0;
        double J = 0.0;
        for(int i = 0; i < h.length; i++) {
            sumOfSquares += Math.pow((y[i] - h[i]), 2);
        }
        J = (1 / (2 * h.length)) * sumOfSquares;
        return J;
    }
}
