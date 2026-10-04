import cv2, numpy as np

badges = [(614,544),(614,702)]

def find_peak(img, tpl, region):
    x0,y0,x1,y1 = region
    sub = img[y0:y1, x0:x1]
    r = cv2.matchTemplate(sub, tpl, cv2.TM_CCOEFF_NORMED)
    mn,mx,ml,ml2 = cv2.minMaxLoc(r)
    return (x0+ml2[0], y0+ml2[1], float(mx))


if __name__ == "__main__":
    # プローブ本体（ローカル実行専用）。
    # import 時に gitignore 対象の画像を読むと、CI の fresh checkout では
    # cv2.imread が None を返し cv2.cvtColor が TypeError になるため、
    # find_peak だけを import 可能にして、I/O はここに閉じ込める。
    f = cv2.imread("output/preview/f01_0182.png")
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

    i_t = cv2.cvtColor(cv2.imread("template/i_icon.png"), cv2.COLOR_BGR2GRAY)
    f_t = cv2.cvtColor(cv2.imread("template/fan_count_label.png"), cv2.COLOR_BGR2GRAY)

    for (bx,by) in badges:
        reg = (bx, by-10, bx+700, by+50)
        print(f"badge=({bx},{by}) i_icon ->", find_peak(g, i_t, reg))
        reg2 = (bx, by+30, bx+500, by+170)
        print(f"badge=({bx},{by}) fan_label ->", find_peak(g, f_t, reg2))
