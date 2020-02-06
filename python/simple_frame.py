# -*- coding: utf-8 -*-

import tkinter as tk


class Application(tk.Frame):
    '''
    デスクトップのアプリのウィンドウを作成するサンプルコード
    '''

    def __init__(self, master=None):
        super().__init__(master)
        master.title("タイトル")
        master.geometry("350x150")
        self.pack()
        self._create_widgets()

    def _create_widgets(self):
        self.lb = tk.Label()
        self.lb["text"] = "ラベルの名前"
        self.lb.pack(side="top")

        self.bt = tk.Button()
        self.bt["text"] = "ボタン"
        self.bt["command"] = self._print_text  # 関数をボタンのクリックに紐付ける
        self.bt.pack(side="bottom")

    def _print_text(self):
        print("ボタンが押された")
        print(self.bt["text"])


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
