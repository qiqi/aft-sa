"""Hand-written spoken forms.

INLINE: keyed by the LaTeX of the inline expression (whitespace-normalized).
DISPLAY: one entry per display equation, in document order, as
         (a substring that must occur in that equation's LaTeX, spoken text).

{a} marks the symbol a, so the synthesizer says the letter name.
build_script.py fails if an expression needs an entry and has none, or if an
entry here matches nothing in the paper.
"""

INLINE = {
    # ---- simple fractions that read better as words
    r'1/2': 'one half',
    r'1/4': 'one quarter',
    r'1/6': 'one sixth',
    r'1/12': 'one twelfth',
    r'1/24': 'one twenty-fourth',
    r'1/48': 'one forty-eighth',
    r'8/8': '8 out of 8',
    r'12/12': '12 out of 12',
    r'c_{\nu,\mathrm{ai}}=1/6': 'c sub nu comma {a} i equals one sixth',
    r'c_{\nu,\mathrm{ai}}=\tfrac16': 'c sub nu comma {a} i equals one sixth',
    # ---- order, functions, derivatives
    r'O(1)': 'order 1',
    r'\hat I=O(d^3)': 'capital I hat is of order d cubed',
    r"I\approx-\tfrac18u''''_w\,d^4":
        'capital I is approximately minus one eighth of u quadruple prime sub w, times d to the '
        'power 4',
    r"U_\mathrm{loc}\approx\sqrt2\,|u'_w|\,d":
        'capital U sub loc is approximately the square root of 2, times the magnitude of u prime '
        'sub w, times d',
    r'\tfrac12 s^2': 'one half s squared',
    r'(1-\sigma_{v1})f_{v1}+\sigma_{v1}':
        'the quantity 1 minus sigma sub v 1, times f sub v 1, plus sigma sub v 1',
    r'O(Re_\theta/Re_L)': 'order R e sub theta over R e sub capital L',
    r'Re_{\theta0}(H)': 'R e sub theta 0, of capital H',
    r'Re_{\theta 0}(H)': 'R e sub theta 0, of capital H',
    r'\max_s\chi(\theta)': 'the maximum over s of chi, as a function of theta',
    r'\rho\tilde\nu f_{v1}(\chi)': 'rho times nu tilde times f sub v 1 of chi',
    r'dN/dRe_\theta': 'd capital N over d R e sub theta',
    r'dN/dRe_\theta\approx10^{-2}':
        'd capital N over d R e sub theta is approximately 10 to the power minus 2',
    r'Re_{\theta0}+1/(dN/dRe_\theta)':
        'R e sub theta 0, plus 1 over the quantity d capital N over d R e sub theta',
    r'N_\mathrm{crit}=\ln(c_{v1}/\chi_\infty)':
        'capital N sub crit equals the natural log of the quantity c sub v 1 over chi sub infinity',
    # ---- roots and powers
    r'\sqrt2': 'the square root of 2',
    r'1/\sqrt{Re_L}': '1 over the square root of R e sub capital L',
    r'\sqrt{c_{\nu,\mathrm{ai}}}': 'the square root of c sub nu comma {a} i',
    r'Y/\sqrt{X^2+Y^2}':
        'capital Y over the square root of the quantity capital X squared plus capital Y squared',
    r'\sqrt{X^2+Y^2+Z^2}':
        'the square root of the quantity capital X squared plus capital Y squared plus capital Z squared',
    r"U_\mathrm{loc}=\sqrt{u^2+(du')^2+(\tfrac12 d^2u'')^2}":
        'capital U sub loc equals the square root of the sum of three squares: u squared; '
        'the quantity d times u prime, squared; and the quantity one half d squared times '
        'u double prime, squared',
    r'G=Re_\delta\sqrt{\delta/R}':
        'capital G equals R e sub delta, times the square root of the quantity delta over capital R',
    r'f^{\,1-g}': 'f to the power of the quantity 1 minus g',
    r'(\tilde\nu/d)^2': 'the quantity nu tilde over d, squared',
    r'g=1-e^{-(\chi-c_{v1})/4}':
        'g equals 1 minus e to the power of: minus the quantity chi minus c sub v 1, all over 4',
    r'B^{n}_i': 'capital B superscript n sub i',
    r"\tfrac12 d^2u''": 'one half d squared times u double prime',
    # ---- fractions with compound parts
    r'A+B/P_I^2': 'capital A, plus capital B over capital P sub capital I squared',
    r'A+B_{\mathrm c}/P_{\mathrm c}^2':
        'capital A, plus capital B sub c over capital P sub c squared',
    r'x^2/A^2+r^2/B^2=1':
        'x squared over capital A squared, plus r squared over capital B squared, equals 1',
    r'd^2\omega/\nu': 'd squared times omega, over nu',
    r'Re_\Omega=d^2\omega/\nu': 'R e sub capital omega equals d squared times omega, over nu',
    r'u=c_1y+c_2y^2': 'u equals c sub 1 times y, plus c sub 2 times y squared',
    r'\chi\!=\!\tilde\nu/\nu\!\sim\!1': 'chi equals nu tilde over nu, of order 1',
    r'f_{v1}=\chi^3/(\chi^3+c_{v1}^3)':
        'f sub v 1 equals chi cubed, over the quantity chi cubed plus c sub v 1 cubed',
    r'(1+c_{b2})\kappa^2u_\tau^2/\sigma':
        'the quantity 1 plus c sub b 2, times kappa squared times u sub tau squared, all over sigma',
    r'c_{b1}/(\kappa^2c_{w1})\approx0.249':
        'c sub b 1, over the quantity kappa squared times c sub w 1, is approximately 0.249',
    r'(1+c_{b2})/(\sigma c_{w1})\approx0.751':
        'the quantity 1 plus c sub b 2, all over the quantity sigma times c sub w 1, '
        'is approximately 0.751',
    r'C_f=2[(\ln Re_\theta)/0.38+3.7]^{-2}':
        'capital C sub f equals 2 times a bracket to the power minus 2. Inside the bracket: '
        'the natural log of R e sub theta, over 0.38, plus 3.7',
    r'18.3\,(\ell/L)': '18.3 times the quantity script ell over capital L',
    r'\delta Z=\pm(d/2R)Y':
        'delta capital Z equals plus or minus the quantity d over 2 capital R, times capital Y',
    r'\delta(\hat\Omega\hat I)/\hat\Omega\hat I':
        'delta of the quantity capital omega hat times capital I hat, '
        'over capital omega hat times capital I hat',
    # ---- magnitudes
    r'|\mathbf{u}|/U_\infty': 'the magnitude of vector u, over capital U sub infinity',
    r'|C_L|<2\times10^{-6}':
        'the magnitude of capital C sub capital L is less than 2 times 10 to the power minus 6',
    r'|C_L|\!<\!2\times10^{-6}':
        'the magnitude of capital C sub capital L is less than 2 times 10 to the power minus 6',
    r'\delta^*\!=\!\nu/|v_w|': 'delta star equals nu over the magnitude of v sub w',
    r'|v_w|/U_\infty\!=\!10^{-2}':
        'the magnitude of v sub w, over capital U sub infinity, equals 10 to the power minus 2',
    r'Re_{\delta^*}\!=\!U_\infty/|v_w|':
        'R e sub delta star equals capital U sub infinity over the magnitude of v sub w',
    r'Re_{\delta^*}\!\approx\!5.4\times10^4':
        'R e sub delta star is approximately 5.4 times 10 to the power 4',
    r'|Y\!-\!X\!-\!Z|/(|X|\!+\!|Y|\!+\!|Z|)':
        'the magnitude of the quantity capital Y minus capital X minus capital Z, all over the sum '
        'of the magnitudes of capital X, capital Y and capital Z',
    # ---- vectors and operators
    r'd\,\nabla d': 'd times the gradient of d',
    r'\mathbf d=d\,\nabla d': 'vector d equals d times the gradient of d',
    r'\nabla^2\mathbf u\!\cdot\!\hat{\mathbf u}':
        'the Laplacian of vector u, dotted with the unit vector u hat',
    r'\nabla^2\mathbf u=-\nabla\times\boldsymbol\omega':
        'the Laplacian of vector u equals minus the curl of vector omega',
    r'\hat{\mathbf n}\!\cdot\!\nabla|\boldsymbol\omega|':
        'the unit vector n hat, dotted with the gradient of the magnitude of vector omega',
    r'\boldsymbol\ell=\tfrac12\langle\mathbf d,\mathbf d\rangle\,\nabla^2\mathbf u':
        'vector script ell equals one half, times the inner product of vector d with vector d, '
        'times the Laplacian of vector u',
    r'\langle-\hat Z\rangle_+': 'the positive part of minus capital Z hat',
    r'a=a_{\max}\operatorname{clip}\langle\hat\Omega\hat I\rangle_0^1':
        '{a} equals {a} sub max, times the clip, between 0 and 1, of capital omega hat times '
        'capital I hat',
    # ---- integrals
    # ---- tuples, intervals, sets
    r'(C,A,B)': 'the triple capital C, capital A, capital B',
    r'(2600,\,175,\,2)': 'the triple 2600, 175, 2',
    r'(g_\chi,g_q)': 'the pair g sub chi, g sub q',
    r'(-8.43,\,2.4)': 'the pair minus 8.43, 2.4',
    r'(A,B)_\mathrm{Mack}': 'the pair capital A, capital B, sub Mack',
    r'(x/L,\phi)': 'the pair x over capital L, phi',
    r'(Re_L,U_\infty)': 'R e sub capital L, capital U sub infinity',
    r'(c_{\nu,\mathrm{ai}},k)=(\tfrac16,\,0.712)':
        'the pair c sub nu comma {a} i, k, equals the pair one sixth, 0.712',
    r'(a_{\mathrm{visc}},B_{\mathrm c})=(0.0276,130)':
        'the pair {a} sub visc, capital B sub c, equals the pair 0.0276, 130',
    r'q\in(2,4)': 'q in the open interval from 2 to 4',
    r'N\!\in\![1,5]': 'capital N in the interval from 1 to 5',
    r'N\!\in\![5,9]': 'capital N in the interval from 5 to 9',
    r'\{0.025,0.2,0.4,0.6,0.8,1.0\}': 'the set 0.025, 0.2, 0.4, 0.6, 0.8, 1.0',
    r'\alpha\in\{4^\circ,5^\circ,6^\circ\}': 'alpha in the set 4 degrees, 5 degrees, 6 degrees',
    r'Re\in\{6\times10^4,\,10^5,\,3\times10^5,\,4.6\times10^5\}':
        'R e in the set 6 times 10 to the power 4, 10 to the power 5, 3 times 10 to the power 5, '
        '4.6 times 10 to the power 5',
    r'\mathbb{RP}^2': 'R P 2',
    # ---- the Daedalus section definition
    r'z_c=x_e[\hat x(1-\hat x)\sum_{i=0}^{10}A^{c}_iB^{10}_i(\hat x)] +\hat x\,\Delta z_{te}':
        'z sub c equals x sub e times a bracket, plus x hat times capital delta z sub t e. '
        'Inside the bracket: x hat, times the quantity 1 minus x hat, times the sum from i equals 0 '
        'to 10 of capital A superscript c sub i, times capital B superscript 10 sub i of x hat',
    r't=x_e[\hat x^{1/2}(1-\hat x)\sum_{i=0}^{9}A^{t}_iB^{9}_i(\hat x) +\hat x\,t_{te}/(2x_e)]':
        't equals x sub e times a bracket. Inside the bracket: x hat to the power one half, times '
        'the quantity 1 minus x hat, times the sum from i equals 0 to 9 of capital A superscript t '
        'sub i, times capital B superscript 9 sub i of x hat; plus x hat times t sub t e, over the '
        'quantity 2 x sub e',
}

DISPLAY = [
    (r'\frac{D\ln\tilde\nu}{Dt}',
     'Equation 1. Capital D by capital D t of the natural log of nu tilde, equals {a} times sigma '
     'sub R e times omega.'),
    (r'\frac{\partial\rho}{\partial t}',
     'Equation 2, first line. Partial rho with respect to t, plus the divergence of the quantity '
     'rho times vector u, equals 0. Next: the partial with respect to t of the quantity rho times '
     'vector u, plus the divergence of the quantity rho times vector u outer product vector u, '
     'equals minus the gradient of p, plus the divergence of tensor tau.'),
    (r'\boldsymbol\tau = (\mu+\mu_t)',
     'Equation 2, second line. Tensor tau equals the quantity mu plus mu sub t, times a bracket. '
     'Inside the bracket: the gradient of vector u, plus the transpose of the gradient of vector u, '
     'minus two thirds times the divergence of vector u times the identity tensor capital I. '
     'Next: mu sub t equals a bracket times rho times nu tilde. Inside the bracket: the quantity '
     '1 minus sigma sub v 1, times f sub v 1 of chi, plus sigma sub v 1.'),
    (r'\frac{D\tilde\nu}{Dt} = P - D',
     'Equation 3. Capital D by capital D t of nu tilde, equals capital P minus capital D, plus 1 '
     'over sigma times the divergence of a bracket, plus c sub b 2 over sigma, times the squared '
     'magnitude of the gradient of nu tilde. Inside the bracket: the quantity c sub nu comma {a} i '
     'times nu, plus nu tilde, times the gradient of nu tilde.'),
    (r'P = \max\!\big[',
     'Equation 4. Capital P equals the maximum of two terms. First term: the quantity 1 minus sigma '
     'sub capital P, times {a} times sigma sub R e times omega times nu tilde. Second term: sigma '
     'sub capital P times capital P sub S A. Next: capital D equals sigma sub capital D times '
     'capital D sub S A.'),
    (r'P_\mathrm{SA} = c_{b1}\tilde S',
     'Equation 5, first line. Capital P sub S A equals c sub b 1 times capital S tilde times nu '
     'tilde. Capital D sub S A equals c sub w 1 times f sub w, times the quantity nu tilde over d, '
     'squared. Capital S tilde equals omega, plus nu tilde over the quantity kappa squared d '
     'squared, times f sub v 2. f sub v 2 equals 1 minus a fraction: chi, over the quantity 1 plus '
     'chi times f sub v 1.'),
    (r'f_{v1} = \frac{\chi^3}{\chi^3+c_{v1}^3}',
     'Equation 5, second line. f sub v 1 equals chi cubed, over the quantity chi cubed plus c sub '
     'v 1 cubed. f sub w equals g times a fraction raised to the power one sixth. The fraction is: '
     '1 plus c sub w 3 to the power 6, all over g to the power 6 plus c sub w 3 to the power 6. '
     'g equals r plus c sub w 2 times the quantity r to the power 6 minus r. r equals the minimum '
     'of two values: nu tilde over the quantity capital S tilde times kappa squared d squared; '
     'and 10.'),
    (r'\sigma_P = \sigma_t(\chi;\tau)',
     'Equation 6. Sigma sub capital P equals sigma sub t of chi, with parameter tau. Sigma sub t '
     'of chi, with parameter tau, equals the maximum of two values: 1 minus e to the power of '
     'minus the quantity chi minus 1, all over tau; and 0. Sigma sub capital D equals 1 minus a '
     'fraction times the quantity 1 minus sigma sub capital P. The fraction is c sub b 1, over the '
     'quantity kappa squared times c sub w 1.'),
    (r"(X,Y,Z) = \big(u,\;d\,u'",
     'Equation 7. The triple capital X, capital Y, capital Z equals the triple: u; d times u '
     'prime; one half d squared times u double prime. Next: the triple capital X hat, capital Y '
     'hat, capital Z hat equals the triple capital X, capital Y, capital Z, divided by capital U '
     'sub loc.'),
    (r'\hat I = \hat Y - \hat X - \hat Z',
     'Equation 8. Capital I hat equals capital Y hat minus capital X hat minus capital Z hat. '
     'Accumulated inflection.'),
    (r'I(d) \equiv U_\mathrm{loc}\,\hat I',
     'Equation 9. Capital I of d, defined as capital U sub loc times capital I hat, equals d times '
     'u prime, minus u, minus one half d squared times u double prime; which equals minus the '
     'integral from 0 to d of one half s squared times u triple prime of s, d s.'),
    (r'\hat\Omega = \frac{\hat Y}{\sqrt{\hat X^2+\hat Y^2}}',
     'Equation 10. Capital omega hat equals capital Y hat, over the square root of the quantity '
     'capital X hat squared plus capital Y hat squared; which equals capital Y, over the square '
     'root of the quantity capital X squared plus capital Y squared. Vorticity fraction.'),
    (r'\langle\mathbf u,\mathbf u\rangle,\; \sqrt{',
     'Equation 11. The triple capital X, capital Y, capital Z equals three entries. First entry: '
     'the inner product of vector u with vector u. Second entry: the square root of the product of '
     'three inner products: vector u with vector u, vector d with vector d, and vector omega with '
     'vector omega. Third entry: the inner product of vector script ell with vector u. Next: '
     'capital omega hat equals capital Y, over the square root of the quantity capital X squared '
     'plus capital Y squared. And capital I hat equals the quantity capital Y minus capital X '
     'minus capital Z, all over the square root of the quantity capital X squared plus capital Y '
     'squared plus capital Z squared.'),
    (r'\big(\,|\mathbf u|,\;\; d\,|\boldsymbol\omega|',
     'Equation 12. The triple capital X, capital Y, capital Z equals three entries. First entry: '
     'the magnitude of vector u. Second entry: d times the magnitude of vector omega. Third entry: '
     'one half d squared, times the Laplacian of vector u, dotted with the unit vector u hat.'),
    (r'P_{\mathrm{AI}} = a\,\sigma_{Re}\,\omega\,\tilde\nu',
     'Equation 13. Capital P sub A I equals {a} times sigma sub R e times omega times nu tilde. '
     'Next: {a} equals {a} sub max, times the clip, between 0 and 1, of capital omega hat times '
     'capital I hat. Next: sigma sub R e equals the function sigma sub R e, evaluated at a ratio: '
     'R e sub capital omega, over R e sub capital omega superscript c of capital omega hat times '
     'capital I hat. And the function sigma sub R e of zeta equals one half times a bracket: 1 '
     'plus the hyperbolic tangent of the quantity zeta minus 1, all over w.'),
    (r'Re_\Omega^{\mathrm c}(\hat\Omega\hat I) = k\,\operatorname{softmin}_2',
     'Equation 14. R e sub capital omega superscript c, as a function of capital omega hat times '
     'capital I hat, equals k times soft-min sub 2 of two arguments. First argument: capital C. '
     'Second argument: capital A, plus capital B times the positive part of capital omega hat '
     'times capital I hat, raised to the power minus 2. Next: soft-min sub 2 of x sub 1 and x sub '
     '2 equals x sub 1 times x sub 2, over the square root of the quantity x sub 1 squared plus x '
     'sub 2 squared. And the positive part of x, written x in angle brackets sub plus, equals the '
     'maximum of x and 0.'),
    (r'c_{\nu,\mathrm{ai}}=\tfrac16 .',
     'c sub nu comma {a} i equals one sixth.'),
    (r'q = \frac{|\mathbf u|\,d/\nu}',
     'Equation 15. q equals a fraction. Numerator: the magnitude of vector u, times d, over nu. '
     'Denominator: y plus, times capital U plus of y plus. Next: capital U plus of y plus equals 1 '
     'over kappa times the natural log of the quantity 1 plus kappa times y plus; plus 7.8 times a '
     'bracket. Inside the bracket: 1, minus e to the power of minus y plus over 11, minus y plus '
     'over 11 times e to the power of minus y plus over 3. Reichardt.'),
    (r'\sigma_{v1} = g_\chi(\chi)\,g_q(q)',
     'Equation 16. Sigma sub v 1 equals g sub chi of chi, times g sub q of q. Next: g sub chi '
     'equals the clip, between 0 and 1, of chi minus 1. And g sub q equals the clip, between 0 and '
     '1, of the quantity q minus 2, all over 2.'),
    (r'N_\mathrm{crit}(Tu) = -8.43',
     'Equation 17. Capital N sub crit, as a function of T u, equals minus 8.43, minus 2.4 times '
     'the natural log of T u sub frac. Next: chi sub infinity equals c sub v 1 times e to the '
     'power of minus capital N sub crit.'),
    (r'c_{b1}=0.1355',
     'c sub b 1 equals 0.1355. Sigma equals two thirds. c sub b 2 equals 0.622. Kappa equals 0.41. '
     'c sub w 1 equals c sub b 1 over kappa squared, plus the quantity 1 plus c sub b 2, all over '
     'sigma. c sub w 2 equals 0.3. c sub w 3 equals 2. c sub v 1 equals 7.1.'),
    (r'P_{\mathrm{c}} \;=\; \hat\Omega',
     'Equation 18. Capital P sub c equals capital omega hat times the positive part of minus '
     'capital Z hat. Next: capital Z hat equals capital Z over capital U sub loc.'),
    (r'\operatorname{softmax}_2\!\Big[',
     'Equation 19, first line. Capital P sub A I equals omega times nu tilde, times soft-max sub 2 '
     'of two arguments. First argument: {a} sub max times capital P sub capital I, times sigma sub '
     'R e of the ratio R e sub capital omega over R e sub capital omega superscript c comma i. '
     'Second argument: {a} sub visc times capital P sub c, times sigma sub R e of the ratio R e '
     'sub capital omega over R e sub capital omega superscript c comma v.'),
    (r'Re_\Omega^{\mathrm c,i} = k\operatorname{softmin}_2',
     'Equation 19, second line. R e sub capital omega superscript c comma i equals k times '
     'soft-min sub 2 of two arguments: capital C; and capital A plus capital B times capital P sub '
     'capital I to the power minus 2. Next: R e sub capital omega superscript c comma v equals '
     'capital A, plus capital B sub c times capital P sub c to the power minus 2. And soft-max sub '
     '2 of x and y equals the square root of the quantity x squared plus y squared.'),
]
