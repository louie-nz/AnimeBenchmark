           generator  n_ref    S         CI95    I    P    Q  S_L1  S_L2  S_TaskB  S_noref  ref_gain  S_corrected
           gptimage2     80 96.5 [94.4, 98.4] 3.88 3.84 3.92  96.1  95.1     99.4     76.3      20.4         91.1
           qwenimage     81 91.6 [89.2, 93.6] 3.40 3.79 3.91  92.4  90.0     90.1     69.1      22.9         86.7
        flux2klein9b     81 90.5 [87.8, 93.0] 3.38 3.75 3.85  90.2  91.3     90.9     78.1      12.5         85.6
        flux2klein4b     81 88.7 [84.4, 92.6] 3.33 3.65 3.80  88.1  89.8     89.6     73.8      15.2         83.8
             sd35ipa     81 66.5 [59.1, 72.8] 2.75 2.22 3.86  68.7  68.5     56.7     62.9       4.0         62.4
anythingv5_ipadapter     81 47.7 [38.3, 56.6] 2.69 1.30 3.84  56.0  60.1      7.1     38.3       9.9         44.4

Pairs not significant after Holm correction are ties.

                   A                    B  n_groups  diff_A_minus_B        p   p_holm  significant
anythingv5_ipadapter         flux2klein9b        27           -42.8 0.000006 0.000084         True
             sd35ipa            gptimage2        27           -30.0 0.000006 0.000084         True
anythingv5_ipadapter            gptimage2        27           -48.8 0.000006 0.000084         True
anythingv5_ipadapter            qwenimage        27           -43.9 0.000006 0.000084         True
           qwenimage              sd35ipa        27            25.1 0.000006 0.000084         True
        flux2klein9b              sd35ipa        27            24.1 0.000011 0.000110         True
        flux2klein4b anythingv5_ipadapter        27            41.0 0.000012 0.000110         True
        flux2klein4b              sd35ipa        27            22.2 0.000016 0.000127         True
        flux2klein9b            gptimage2        27            -6.0 0.000503 0.003523         True
anythingv5_ipadapter              sd35ipa        27           -18.8 0.001203 0.007216         True
           qwenimage            gptimage2        27            -4.9 0.002263 0.011315         True
        flux2klein4b            gptimage2        27            -7.8 0.003109 0.012438         True
        flux2klein4b         flux2klein9b        27            -1.8 0.148564 0.445692        False
        flux2klein4b            qwenimage        27            -2.9 0.573052 1.000000        False
        flux2klein9b            qwenimage        27            -1.1 0.831264 1.000000        False