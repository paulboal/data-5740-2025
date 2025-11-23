
from manim import *
import math

class SimpleT(Scene):
    def construct(self):
        # ---------- PARAMETERS YOU CAN TWEAK ----------
        MU1_START = -1
        MU1_END = 0
        MU2_START = 1
        MU2_END = 0
        SIGMA_START = 1
        SIGMA_END = 2.2
        # ------------------------------------------------

        # Axes for the curves
        axes = Axes(
                    x_range=[-6, 6, 1],
                    y_range=[0, 1.2, 0.2],
                    x_length=10,
                    y_length=3,
                    tips=False,
                ).to_edge(UP)
        x_label = axes.get_x_axis_label("x")
        self.play(Create(axes), FadeIn(x_label))

        # A tracker for sigma so we can animate it
        sigma_tracker = ValueTracker(SIGMA_START)
        mu1_tracker = ValueTracker(MU1_START)
        mu2_tracker = ValueTracker(MU2_START)

        # Normal pdf helper
        def normal_pdf(x, mu, sigma):
            return 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

        # Curves that update with sigma
        def get_curve_mu1():
            sigma = sigma_tracker.get_value()
            MU1 = mu1_tracker.get_value()
            curve = axes.plot(
                lambda x: normal_pdf(x, MU1, sigma),
                x_range=[-6, 6],
                color=BLUE,
                stroke_width=4,
            )
            return curve

        def get_area_mu1():
            area = axes.get_area(get_curve_mu1(), x_range=[-6,6], color=BLUE, opacity=0.5)
            return area


        def get_curve_mu2():
            sigma = sigma_tracker.get_value()
            MU2 = mu2_tracker.get_value()
            return axes.plot(
                lambda x: normal_pdf(x, MU2, sigma),
                x_range=[-6, 6],
                color=YELLOW,
                stroke_width=4,
            )
        
        def get_area_mu2():
            area = axes.get_area(get_curve_mu2(), x_range=[-6,6], color=YELLOW, opacity=0.5)
            return area


        curve1 = always_redraw(get_curve_mu1)
        area1 = always_redraw(get_area_mu1)
        curve2 = always_redraw(get_curve_mu2)
        area2 = always_redraw(get_area_mu2)

        self.play(Create(curve1), FadeIn(area1), Create(area2),  FadeIn(curve2))

        # ---------- Z-SCORE AND P-VALUE DISPLAY ----------
        # z = |mu2 - mu1| / sigma
        # p = 2 * (1 - Phi(z))
        # Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))

        def phi(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        z_text = always_redraw(
            lambda: MathTex(
                r"z = \frac{|" + f"{mu2_tracker.get_value()} - {mu1_tracker.get_value()}" + r"|}{\sigma} = " + f"{abs(mu2_tracker.get_value() - mu1_tracker.get_value())/sigma_tracker.get_value():.2f}"
            ).scale(0.7).to_corner(DOWN + LEFT)
        )

        p_text = always_redraw(
            lambda: MathTex(
                r"p = " + f"{2*(1 - phi(abs(mu2_tracker.get_value() - mu1_tracker.get_value())/sigma_tracker.get_value())):.3f}"
            ).scale(0.7).next_to(z_text, DOWN, aligned_edge=LEFT)
        )

        sigma_text = always_redraw(
            lambda: MathTex(
                r"\sigma = " + f"{sigma_tracker.get_value():.2f}"
            ).scale(0.7).to_corner(DOWN + RIGHT)
        )

        self.play(FadeIn(z_text), FadeIn(p_text), FadeIn(sigma_text))

        # ---------- ANIMATE: sigma from small → large ----------
        self.play(
            # sigma_tracker.animate.set_value(SIGMA_END),
            mu1_tracker.animate.set_value(MU1_END),
            run_time=3,
            rate_func=rate_functions.linear
        )

        self.play(
            # sigma_tracker.animate.set_value(SIGMA_END),
            mu2_tracker.animate.set_value(MU2_END),
            run_time=3,
            rate_func=rate_functions.linear
        )

        self.wait(2)