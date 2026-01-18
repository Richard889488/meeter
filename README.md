# Meeter – Real-Time Camera Interaction Tool

Meeter is a Windows desktop application designed for **real-time camera interaction, face-based gameplay, and experimental human–computer interaction scenarios**.  
It is primarily intended for **research, education, prototyping, and non-commercial demonstrations**.

This project integrates computer vision, local server control, and real-time interaction logic into a single executable, allowing end users to run the system **without installing Python or development tools**.

---

## 📌 Purpose of This Software

Meeter was created to explore and demonstrate:

- Real-time camera input processing
- Face-based interaction and recognition workflows
- Experimental shooting / hit-detection mechanics
- Human–computer interaction concepts for games, exhibitions, and research demos
- Educational demonstrations of AI-assisted interaction systems

Typical use cases include:

- Academic or science-fair projects  
- Classroom or workshop demonstrations  
- Prototype interactive games or installations  
- Personal experimentation with computer vision systems  

---

## 🖥️ How to Use (End Users)

1. Download the Windows executable from the **Releases** section.
2. Double-click the `.exe` file to launch.
3. Allow camera access when prompted.
4. Open the local interface in your browser if instructed by the application.
5. Follow on-screen instructions to interact with the system.

> ⚠️ Windows may show a security warning because this application is not code-signed.  
> Choose **“More info → Run anyway”** if you trust the source.

---

## 🧠 Technical Overview (For Developers)

- Desktop application packaged as a standalone Windows executable
- Uses local camera input for real-time processing
- Integrates face analysis and interaction logic
- Local Flask server for UI and control flow
- Designed to run entirely on the user’s machine (no cloud AI dependency required at runtime)

This repository **does not aim to be a reusable commercial SDK**, but rather a **transparent reference implementation** for learning and experimentation.

---

## 🚫 Non-Commercial Use Only (Important)

Although this project is released under the **MIT License**, the author **explicitly restricts its use to non-commercial purposes**.

### You MAY:
- Use this software for learning, research, education, and personal projects
- Modify the source code for non-commercial experimentation
- Share the software for academic or demonstration purposes

### You MAY NOT:
- Sell this software or any derivative work
- Bundle it into a paid product or service
- Use it in commercial installations, products, or monetized platforms
- Redistribute modified versions for profit

If you are interested in **commercial usage, licensing, or collaboration**, you must contact the author for explicit permission.

---

## ⚠️ Disclaimer

This software is provided **“as is”**, without warranty of any kind.

- The author is not responsible for misuse, data loss, or unintended consequences.
- Camera data is processed locally; however, users are responsible for complying with local privacy and data protection laws.
- This project is **not intended for surveillance or identification of individuals without consent**.

---

## 📄 License

MIT License  
Copyright © 2026  

> Additional non-commercial usage restrictions apply.  
> See the section **“Non-Commercial Use Only”** above.

---

## 🌐 Project Information

This project is part of experimental research and development under the  
**RIGHTLEFTUP** initiative.

Website: https://rightleftup.com  

---

## 🙏 Acknowledgments

This project builds upon open-source libraries and the broader computer vision research community.  
Thanks to all contributors and open-source maintainers who make experimentation possible.
