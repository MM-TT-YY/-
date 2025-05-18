import pygame
import random
import numpy as np
import sounddevice as sd
import threading
import queue
from pygame import freetype
import os
import cv2

# 窗口初始化
pygame.init()
WIDTH, HEIGHT = 450, 800
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("耄耋大狗嚼")
icon = pygame.image.load("maodie.png")
pygame.display.set_icon(icon)
clock = pygame.time.Clock()

# 字体
freetype.init()
font = freetype.SysFont("SimHei", 24)
font_bold48 = freetype.SysFont("Microsoft YaHei", 46, bold=True)
font_score = freetype.SysFont("SimHei", 80, bold=True)

# 图像资源
maodie_img = pygame.image.load("maodie.png").convert_alpha()
maodie_img = pygame.transform.scale(maodie_img, (60, 60))

dagou_img_raw = pygame.image.load("dagou.png").convert_alpha()
DAGOU_WIDTH = 100
gap = 200  # 大狗之间的间距

# 得分图片
huotuichang_img_raw = pygame.image.load("huotuichang.png").convert_alpha()
score_font_height = font_score.get_sized_height()
HUOTUICHANG_HEIGHT = int(score_font_height * 1.18)
w, h = huotuichang_img_raw.get_size()
huotuichang_scale = HUOTUICHANG_HEIGHT / h
HUOTUICHANG_SIZE = (int(w * huotuichang_scale), HUOTUICHANG_HEIGHT)
huotuichang_img = pygame.transform.smoothscale(huotuichang_img_raw, HUOTUICHANG_SIZE)

# 死亡音效
dead_sound_path = os.path.join(os.path.dirname(__file__), "dead.mp3")
dead_sound = None
if os.path.exists(dead_sound_path):
    dead_sound = pygame.mixer.Sound(dead_sound_path)

# 声音采集
volume_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    volume_norm = np.linalg.norm(indata) * 10
    volume_queue.put(volume_norm)

stream = sd.InputStream(callback=audio_callback)
stream.start()

# 哈气特效
class FadeText:
    def __init__(self, text, pos, font, color=(255, 255, 255), duration=60, float_speed=-3):
        self.text = text
        self.pos = list(pos)
        self.font = font
        self.color = color
        self.duration = duration
        self.alpha = 255
        self.surface, _ = font.render(text, color)
        self.surface = self.surface.convert_alpha()
        self.float_speed = float_speed  # 上浮速度

    def update(self):
        if self.alpha > 0:
            self.alpha -= 255 // self.duration
            self.surface.set_alpha(max(self.alpha, 0))
            self.pos[1] += self.float_speed  # 上浮

    def draw(self, win):
        if self.alpha > 0:
            win.blit(self.surface, self.pos)

# 血液粒子
class BloodParticle:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.radius = random.randint(2, 5)
        self.color = (139, 0, 0)
        self.vel = [random.uniform(-2, 2), random.uniform(-2, 2)]
        self.lifetime = 90  # 延长寿命
        self.gravity = 0.25

    def update(self):
        self.x += self.vel[0]
        self.y += self.vel[1]
        self.vel[1] += self.gravity  # 加入重力
        self.lifetime -= 1

    def draw(self, win):
        if self.lifetime > 0:
            pygame.draw.circle(win, self.color, (int(self.x), int(self.y)), self.radius)

# 耄耋类
class Maodie:
    def __init__(self):
        self.x, self.y = 100, HEIGHT // 2
        self.velocity = 0
        self.gravity = 0.4
        self.lift = -7
        self.radius = int(30 * 0.8)  # 实际判定点是0.8倍半径
        self.angle = 0    # 当前旋转角度
        self.angle_target = 0  # 目标角度
        self.angle_speed = 0.5 # 旋转变化速度

    def update(self, volume_triggered):
        if volume_triggered:
            self.velocity = self.lift
        else:
            self.velocity += self.gravity
        self.y += self.velocity
        # 边缘反弹
        if self.y < 0:
            self.y = 0
            self.velocity = -self.velocity * 0.7
        if self.y > HEIGHT - 60:
            self.y = HEIGHT - 60
            self.velocity = -self.velocity * 0.7
        if abs(self.angle - self.angle_target) < 0.1:
            self.angle_target = random.uniform(-5, 5)
        self.angle += (self.angle_target - self.angle) * self.angle_speed

    def draw(self, win):
        # 随机旋转
        rotated_img = pygame.transform.rotate(maodie_img, self.angle)
        rect = rotated_img.get_rect(center=(self.x + self.radius, self.y + self.radius))
        win.blit(rotated_img, rect.topleft)
        # 可视化圆形碰撞体
        # pygame.draw.circle(win, (0,255,0), (self.x + self.radius, self.y + self.radius), self.radius, 2)

    def get_circle(self):
        # 返回圆心和半径
        return (self.x + self.radius, self.y + self.radius, self.radius)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, 60, 60)

# 大狗类
class BigDog:
    def __init__(self, x):
        self.x = x
        self.width = DAGOU_WIDTH
        self.dog_img = pygame.transform.scale(dagou_img_raw, (DAGOU_WIDTH, int(dagou_img_raw.get_height() * DAGOU_WIDTH / dagou_img_raw.get_width())))
        self.image_height = self.dog_img.get_height()
        self.top = random.randint(200, HEIGHT - 200)
        self.passed = False
        # 随机偏移量
        self.offset_x = random.uniform(-6, 6)
        self.offset_y = random.uniform(-6, 6)
        self.offset_phase = random.uniform(0, 2 * np.pi)
        self.offset_speed = random.uniform(0.005, 0.012)
        # 随机慢速微小旋转
        self.angle = 0
        self.angle_target = random.uniform(-1, 1)
        self.angle_speed = 0.1

    def update(self):
        self.x -= 2
        # 缓慢变化的抖动
        t = pygame.time.get_ticks()
        self.offset_x = 6 * np.sin(self.offset_phase + t * self.offset_speed)
        self.offset_y = 6 * np.cos(self.offset_phase + t * self.offset_speed)
        # 慢速微小旋转
        if abs(self.angle - self.angle_target) < 0.05:
            self.angle_target = random.uniform(-2, 2)
        self.angle += (self.angle_target - self.angle) * self.angle_speed

    def draw(self, win):
        # 上狗
        top_img = pygame.transform.rotate(self.dog_img, 180 + self.angle)
        win.blit(top_img, (self.x + self.offset_x, self.top - gap - self.image_height + self.offset_y))
        # 下狗
        dog_img_rot = pygame.transform.rotate(self.dog_img, self.angle)
        win.blit(dog_img_rot, (self.x + self.offset_x, self.top + self.offset_y))

    def collide(self, maodie_circle):
        # 主矩形（碰撞区域）加上偏移
        bottom_rect = pygame.Rect(self.x + self.offset_x, self.top + 60 + self.offset_y, self.width, HEIGHT - self.top - 30)
        top_rect = pygame.Rect(self.x + self.offset_x, 0 + self.offset_y, self.width, self.top - gap - 60)

        # 三角形尖角（碰撞区域）加上偏移
        triangle1 = [
            (self.x + self.offset_x, self.top + 60 + self.offset_y),
            (self.x + self.width//2 + self.offset_x, self.top + self.offset_y),
            (self.x + self.width + self.offset_x, self.top + 60 + self.offset_y)
        ]
        triangle2 = [
            (self.x + self.offset_x, self.top - gap - 60 + self.offset_y),
            (self.x + self.width//2 + self.offset_x, self.top - gap + self.offset_y),
            (self.x + self.width + self.offset_x, self.top - gap - 60 + self.offset_y)
        ]
        if circle_rect_collision(maodie_circle, bottom_rect) or circle_rect_collision(maodie_circle, top_rect):
            return True
        if circle_triangle_collision(maodie_circle, triangle1) or circle_triangle_collision(maodie_circle, triangle2):
            return True
        return False

# 圆形与矩形碰撞检测
def circle_rect_collision(circle, rect):
    cx, cy, cr = circle
    rx, ry, rw, rh = rect
    # 找到圆心到矩形最近点
    nearest_x = max(rx, min(cx, rx + rw))
    nearest_y = max(ry, min(cy, ry + rh))
    dx = cx - nearest_x
    dy = cy - nearest_y
    return dx * dx + dy * dy <= cr * cr

# 圆形与三角形碰撞检测（简单近似：圆心在三角形内或圆心到三角形边距离小于半径）
def circle_triangle_collision(circle, triangle):
    cx, cy, cr = circle
    # 圆心在三角形内
    if point_in_triangle((cx, cy), triangle):
        return True
    # 检查圆与三角形三条边的最短距离
    for i in range(3):
        p1 = triangle[i]
        p2 = triangle[(i+1)%3]
        if point_to_segment_dist((cx, cy), p1, p2) <= cr:
            return True
    return False

def point_to_segment_dist(p, a, b):
    # 点p到线段ab的最短距离
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    nearest_x = ax + t * dx
    nearest_y = ay + t * dy
    return ((px - nearest_x) ** 2 + (py - nearest_y) ** 2) ** 0.5

def point_in_triangle(p, triangle):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0])*(p2[1] - p3[1]) - (p2[0] - p3[0])*(p1[1] - p3[1])
    b1 = sign(p, triangle[0], triangle[1]) < 0.0
    b2 = sign(p, triangle[1], triangle[2]) < 0.0
    b3 = sign(p, triangle[2], triangle[0]) < 0.0
    return ((b1 == b2) and (b2 == b3))

# 播放背景音乐
pygame.mixer.init()
music_path = os.path.join(os.path.dirname(__file__), "haqi.mp3")
if os.path.exists(music_path):
    pygame.mixer.music.load(music_path)
    pygame.mixer.music.set_volume(0.2)
    pygame.mixer.music.play(-1)

# 播放CG
def play_cg(win, alpha=255):
    cg_path = os.path.join(os.path.dirname(__file__), "cg.mp4")
    if not os.path.exists(cg_path):
        return False
    cap = cv2.VideoCapture(cg_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    fade_duration = 2.0
    fade_start_frame = total_frames - int(fps * fade_duration)

    clock = pygame.time.Clock()
    frame_idx = 0
    running = True
    cg_surfaces = []
    # 预先加载所有帧到内存
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(np.flipud(np.rot90(frame)))
        surf = pygame.transform.scale(surf, (win.get_width(), win.get_height()))
        # 处理淡出
        if frame_idx >= fade_start_frame:
            frame_alpha = int(255 * (total_frames - frame_idx) / (total_frames - fade_start_frame))
            frame_alpha = max(0, min(255, frame_alpha))
        else:
            frame_alpha = 255
        surf.set_alpha(min(frame_alpha, alpha))
        cg_surfaces.append(surf)
    cap.release()
    # 播放所有帧
    for surf in cg_surfaces:
        win.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
    return False

# 预加载CG视频帧
def preload_cg_frames(win):
    cg_path = os.path.join(os.path.dirname(__file__), "cg.mp4")
    cg_frames = []
    cg_fps = 30
    cg_total_frames = 0
    if not os.path.exists(cg_path):
        return cg_frames, cg_fps, cg_total_frames
    cap = cv2.VideoCapture(cg_path)
    cg_fps = cap.get(cv2.CAP_PROP_FPS)
    cg_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(np.flipud(np.rot90(frame)))
        surf = pygame.transform.scale(surf, (win.get_width(), win.get_height()))
        cg_frames.append(surf)
    cap.release()
    return cg_frames, cg_fps, cg_total_frames

def show_title_screen(win):
    # 载入title.png
    title_path = os.path.join(os.path.dirname(__file__), "title.png")
    title_img = None
    if os.path.exists(title_path):
        title_img = pygame.image.load(title_path).convert_alpha()
        title_img = pygame.transform.scale(title_img, (win.get_width(), win.get_height()))
    # 显示标题画面
    clock = pygame.time.Clock()
    showing = True
    cg_frame_surfaces, cg_fps, cg_total_frames = [], 0, 0
    preload_done = False
    preload_thread = None

    def preload():
        nonlocal cg_frame_surfaces, cg_fps, cg_total_frames, preload_done
        cg_frame_surfaces, cg_fps, cg_total_frames = preload_cg_frames(win)
        preload_done = True

    # 启动预加载线程
    preload_thread = threading.Thread(target=preload)
    preload_thread.start()

    while showing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                showing = False

        win.fill((0, 0, 0))
        if title_img:
            win.blit(title_img, (0, 0))
        if not preload_done:
            loading_font = freetype.SysFont("SimHei", 12)
            loading_txt, _ = loading_font.render("正在预加载资源...", (200, 200, 200))
            win.blit(loading_txt, (win.get_width() // 2 - loading_txt.get_width() // 2, win.get_height() - 60))
        else:
            loaded_font = freetype.SysFont("SimHei", 12)
            loaded_txt, _ = loaded_font.render("资源已加载", (255, 255, 255))
            win.blit(loaded_txt, (win.get_width() // 2 - loaded_txt.get_width() // 2, win.get_height() - 60))
        pygame.display.flip()
        clock.tick(30)
    # 等待预加载线程结束
    if preload_thread and preload_thread.is_alive():
        preload_thread.join()
    return cg_frame_surfaces, cg_fps, cg_total_frames

# 主循环
def main():
    cg_frame_surfaces, cg_fps, cg_total_frames = show_title_screen(win)
    cg_fade_total = int(cg_fps * 2.0) if cg_fps else 1
    cg_fade_start = cg_total_frames - cg_fade_total if cg_total_frames > cg_fade_total else 0

    maodie = Maodie()
    dogs = [BigDog(WIDTH + i * 250) for i in range(3)]
    score = 0
    running = True
    game_over = False
    fade_text = None
    blood_particles = []
    played_dead_sound = False

    VOLUME_THRESHOLD = 10
    sustained_volume = 0
    last_volume_triggered = False

    shake_offset = [0, 0]
    shake_velocity = [0, 0]

    zhanbai_path = os.path.join(os.path.dirname(__file__), "zhanbai.png")
    zhanbai_img = None
    if os.path.exists(zhanbai_path):
        zhanbai_img = pygame.image.load(zhanbai_path).convert_alpha()
        zhanbai_img = pygame.transform.scale(zhanbai_img, (win.get_width(), win.get_height()))

    cg_playing = False
    cg_frame_idx = 0
    zhanbai_show = False  # 控制图片是否持续显示
    cg_played = False     # 本次死亡是否已播放CG

    # 降低zhanbai.png饱和度
    def desaturate_surface(surface):
        arr = pygame.surfarray.pixels3d(surface)
        avg = arr.mean(axis=2, keepdims=True)
        arr[:] = (arr * 0.3 + avg * 0.7).astype(arr.dtype)
        del arr
        return surface

    if zhanbai_img:
        zhanbai_img = desaturate_surface(zhanbai_img.copy())

    while running:
        clock.tick(60)
        # 背景色
        if game_over:
            # 死亡后降低饱和度（灰色调）
            bg_color = (150, 180, 200)
        else:
            bg_color = (135, 206, 235)
        win.fill(bg_color)

        # 音量读取
        volume = 0
        while not volume_queue.empty():
            volume = volume_queue.get()
        volume_triggered = volume > VOLUME_THRESHOLD
        if volume_triggered:
            sustained_volume += 1
        else:
            sustained_volume = 0

        # 音量条
        pygame.draw.rect(win, (255, 255, 255), (10, 10, 20, 200), 2)
        filled_height = min(int((volume / 50) * 200), 200)
        pygame.draw.rect(win, (255, 255, 255), (10, 210 - filled_height, 20, filled_height))
        # 阈值红线
        threshold_height = int((VOLUME_THRESHOLD / 50) * 200)
        pygame.draw.line(win, (255, 0, 0), (10, 210 - threshold_height), (30, 210 - threshold_height), 2)

        # 起跳时显示“哈！”
        if volume_triggered and not last_volume_triggered:
            fade_text = FadeText("哈！", (maodie.x + 10, maodie.y - 50), font, (255, 255, 255), duration=30, float_speed=-3)
        last_volume_triggered = volume_triggered

        if fade_text:
            fade_text.update()
            fade_text.draw(win)
            if fade_text.alpha <= 0:
                fade_text = None

        # 更新角色
        if not game_over:
            # 复活时如未播放BGM则重播
            if not pygame.mixer.music.get_busy() and os.path.exists(music_path):
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.set_volume(0.2)
                pygame.mixer.music.play(-1)
            maodie.update(volume_triggered)
            maodie.draw(win)

            maodie_circle = maodie.get_circle()
            collided = False

            for dog in dogs:
                dog.update()
                dog.draw(win)
                if dog.collide(maodie_circle):
                    collided = True
                if not dog.passed and dog.x + dog.width < maodie.x:
                    score += 1
                    dog.passed = True
                    # 显示得分图片
                    win.blit(huotuichang_img, (WIDTH - 150, 10))
            # 碰撞即game_over
            if collided:
                for _ in range(30):
                    blood_particles.append(BloodParticle(maodie.x + maodie.radius, maodie.y + maodie.radius))
                if not game_over:
                    game_over = True
                    played_dead_sound = False  # 重置死亡音效标志
                    cg_played = False          # 死亡时重置CG播放标志
                    cg_playing = False
                    cg_frame_idx = 0
                    zhanbai_show = False
            # 移除并添加新大狗
            if dogs[0].x + dogs[0].width < 0:
                dogs.pop(0)
                dogs.append(BigDog(dogs[-1].x + 250))
        else:
            # 死亡时只播放一次音效
            if not played_dead_sound and dead_sound:
                dead_sound.play()
                played_dead_sound = True
            # 死亡时淡出BGM
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.fadeout(800)

            if cg_frame_surfaces and not cg_played:
                cg_playing = True
                cg_frame_idx = 0
                zhanbai_show = False
                cg_played = True  # 标记本次死亡已播放CG

            if cg_playing and cg_frame_surfaces and cg_frame_idx < len(cg_frame_surfaces):
                surf = cg_frame_surfaces[cg_frame_idx]
                fadein_frames = int(cg_fps * 0.5) if cg_fps else 1
                if cg_frame_idx < fadein_frames:
                    alpha = int(255 * cg_frame_idx / fadein_frames)
                elif cg_frame_idx >= cg_fade_start:
                    alpha = int(255 * (len(cg_frame_surfaces) - cg_frame_idx) / (len(cg_frame_surfaces) - cg_fade_start))
                else:
                    alpha = 255
                surf.set_alpha(max(0, min(255, alpha)))
                win.blit(surf, (0, 0))
                if zhanbai_img and cg_frame_idx >= cg_fade_start:
                    zhanbai_alpha = int(255 * (cg_frame_idx - cg_fade_start) / (len(cg_frame_surfaces) - cg_fade_start))
                    zhanbai_alpha = max(0, min(255, zhanbai_alpha))
                    zhanbai_img.set_alpha(zhanbai_alpha)
                    win.blit(zhanbai_img, (0, 0))
                    if zhanbai_alpha >= 255:
                        zhanbai_show = True
                cg_frame_idx += 1
                if cg_frame_idx >= len(cg_frame_surfaces):
                    cg_playing = False
                    cg_frame_idx = 0
                    if zhanbai_img:
                        zhanbai_img.set_alpha(255)
                        zhanbai_show = True
            elif zhanbai_img and zhanbai_show:
                zhanbai_img.set_alpha(255)
                win.blit(zhanbai_img, (0, 0))

            # 抖动逻辑（全向，快，幅度小）
            if shake_velocity == [0, 0]:
                # 初始化抖动速度，幅度大幅减小
                shake_velocity = [
                    random.uniform(-4, 4),  # x方向
                    random.uniform(-4, 4)   # y方向
                ]
                shake_offset = [0, 0]
            # 阻尼抖动更新
            shake_offset[0] += shake_velocity[0]
            shake_offset[1] += shake_velocity[1]
            shake_velocity[0] *= 0.5  # 更快阻尼
            shake_velocity[1] *= 0.5
            if abs(shake_velocity[0]) < 1:
                shake_velocity[0] = 0
                shake_offset[0] *= 0.1
            if abs(shake_velocity[1]) < 1:
                shake_velocity[1] = 0
                shake_offset[1] *= 0.1

            txt, _ = font_bold48.render("已经...不用再哈气了", (155, 10, 10))
            win.blit(
                txt,
                (
                    WIDTH // 2 - 200 + int(shake_offset[0]),
                    HEIGHT // 2 - 40 + int(shake_offset[1])
                )
            )
            txt1, _ = font.render("> 扣 1 复 活 耄 耋 <", (255, 255, 255))
            win.blit(txt1, (WIDTH//2 - 140, HEIGHT//2 + 50))

        # 血液粒子
        for particle in blood_particles:
            particle.update()
            particle.draw(win)
        blood_particles = [p for p in blood_particles if p.lifetime > 0]

        # 得分图片（等比缩放后）和分数数值
        score_x = WIDTH - 150
        score_y = 10
        win.blit(huotuichang_img, (score_x - 10, score_y - 1))
        win.blit(huotuichang_img, (score_x, score_y))
        txt_score, _ = font_score.render(f"{score}", (255, 255, 255))
        # 分数垂直居中火腿肠
        score_text_y = score_y + (HUOTUICHANG_HEIGHT - txt_score.get_height()) // 2
        win.blit(txt_score, (score_x + HUOTUICHANG_SIZE[0] + 16, score_text_y))

        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if game_over and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    main()
                    return
                if event.key == pygame.K_1:
                    # 复活
                    maodie = Maodie()
                    dogs = [BigDog(WIDTH + i * 250) for i in range(3)]
                    score = 0
                    game_over = False
                    fade_text = None
                    blood_particles = []
                    sustained_volume = 0
                    last_volume_triggered = False
                    played_dead_sound = False
                    shake_offset = [0, 0]
                    shake_velocity = [0, 0]
                    cg_playing = False
                    cg_frame_idx = 0
                    zhanbai_show = False
                    cg_played = False

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
