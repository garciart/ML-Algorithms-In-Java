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

import static java.lang.System.exit;

/**
 *
 * @author Rob Garcia at rgarcia@rgprogramming.com
 */
public class MLAlgorithmsInJava {

    /** USE DOUBLE WRAPPER, NOT PRIMITIVE TYPE! WE NEED TO DO NULL CHECKS! **/
    
    /**
     * x represents the set of independent variables, known as the set of features.
     * For example, the square footage of a house.
     * x[1][n] must always equal 1 to allow the formula to return theta[0][n] (the y-intercept for all lines of x).
     * There must be a row of x values in the x matrix for each element in the y vector.
     */
    public static Double[][] xMatrix = {{1.0, 90.0},
                                        {1.0, 101.0},
                                        {1.0, 1330.0}
    };
    
    /**
     * y represents the dependent variable, known as the output feature.
     * For example, the actual price of a house based on the square footage ($151 per square foot in 2018).
     * There must be an element in the y vector for each row of x values in the x matrix.
     */
    public static Double[] yVector = {24900.0, 338000.0, 6500000.0};

    /**
     * theta represents the slope of each point in the set of features.
     * For example, the slope of each home price in relation to it's square footage.
     * The number of elements in each row of theta must match the number elements in each row of x.
     * theta[0][n] is equal to the y-intercept of the prediction.
     */
    public static Double[][] thetaMatrix = {{1.004579, 5.286822},
                                            {1004.579, 5286.822}
    };
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        System.out.println("Hello, world!");
        System.out.println("\nChecking data...\n");
        // Check that the number of x values equals the number of y values.
        if(xMatrix.length != yVector.length) {
            System.out.println("Check your data: the number of x elements (" + xMatrix.length + ") does not equal the number of y elements (" + yVector.length + ").");
            exit(0);
        }
        // Check that there is a theta value for each value of x (per row). 
        else if(xMatrix[0].length != thetaMatrix[0].length) {
            System.out.println("Check your data: the number of x elements (" + xMatrix[0].length + ") does not equal the number of theta elements (" + thetaMatrix[0].length + ").");
            exit(0);
        }
        else {
            System.out.println("Data OK.");
            System.out.println("\nCalculating the hypotheses (h(x) = \u019F\u2080 + \u019F\u2081X\u2081 + ... + \u019F\u2099X\u2099)...\n");
            // Create a vector to hold each element in a theta matrix column
            Double[] thetaVector = new Double[thetaMatrix[0].length];
            // Create a vector to hold each element in an x matrix column
            Double[] xVector = new Double[xMatrix[0].length];
            // Create a matrix to hold all the h values per theta and x vector (used to compute sum of squares for each vector)
            Double[][] h = new Double[thetaMatrix.length][xMatrix.length];
            /**
             * To get h, you need to apply each theta vector to each vector of x.
             * The number of theta vectors controls the outer loop.
             * Each theta vector must be applied to each x vector,
             * so the inner loop is controlled by the number of vectors in x.
             * Each value in a theta vector must be applied to each value in an x vector,
             * so the innermost loop is controlled by the number of elements
             * in each theta vector or x vector
             * (the number of elements in each must be the same for matrix multiplication).
            */
            for(int i = 0; i < thetaMatrix.length; i++) {
                for(int j = 0; j < xMatrix.length; j++) {
                    for(int k = 0; k < xMatrix[0].length; k++) {
                        // Create the vectors for matrix multiplication
                        xVector[k] = xMatrix[j][k];
                        thetaVector[k] = thetaMatrix[i][k];
                    }
                    // Apply the hypothesis function. If it fails, it wil return null and the application will exit.
                    if((h[i][j] = LinearRegression.Hypothesis(thetaVector, xVector)) == null) {
                        exit(0);
                    }
                    else {
                        // My fun code to print out each hypothesis function! 
                        System.out.printf("h\u019F(x) = %.2f = ", h[i][j]);
                        for(int k2 = 0; k2 < xVector.length; k2++) {
                            System.out.printf("(%.2f * %.2f)", thetaVector[k2], xVector[k2]);
                            System.out.print(k2 == (xVector.length - 1) ? "" : " + ");
                        }
                        System.out.println();
                    }
                }
                System.out.println();
            }
            System.out.println("\nCalculating the cost [J(\u019F\u2080, \u019F\u2081...\u019F\u2099)) = 1 / (2 * m) * (\u2211((h(x\u2099) - y\u2099)\u00B2))]...\n");
            Double[] J = new Double[thetaMatrix.length];
            for(int i = 0; i < thetaMatrix.length; i++) {
                if((J[i] = LinearRegression.CostFunction(h[i], yVector)) == null) {
                        exit(0);
                }
                else {
                    // My fun code to print out the results of the cost function! 
                    System.out.print("J(");
                    for(int k2 = 0; k2 < thetaMatrix[0].length; k2++) {
                        System.out.printf("%.2f", thetaMatrix[i][k2]);
                        System.out.print(k2 == (thetaMatrix.length - 1) ? "" : ", ");
                    }
                    System.out.printf(") = %.2f\n", J[i]);
                    System.out.println();
                }
            }
            System.out.println("\nPerforming gradient descent...\n");
            Double alpha = 0.0;
            for(int i = 0; i < thetaMatrix.length; i++) {
                for(int j = 0; j < xMatrix[0].length; j++) {
                    for(int k = 0; k < yVector.length; k++) {
                    
                        System.out.println("Theta" + j + " = " + thetaMatrix[i][j]);
                        // System.out.println("x" + j + " = " + xMatrix[j][k]);
                        System.out.println("h\u019F(x) = " + h[j][k]);
                        /** LEFT OFF HERE! WATCH VIDEOS FIRST! **/
                    }
                }
                System.out.println();
            }
        }
    }
}
