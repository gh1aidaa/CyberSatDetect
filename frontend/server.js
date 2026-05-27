import express from "express";
import helmet from "helmet";
import cors from "cors";
import rateLimit from "express-rate-limit";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import validator from "validator";
import xss from "xss";
import "dotenv/config";

const app = express();
app.use(express.json({ limit: "50kb" }));
app.use(helmet());

app.use(cors({
  origin: ["http://127.0.0.1:5500", "http://localhost:5500"],
  methods: ["GET", "POST"],
}));

const authLimiter = rateLimit({
  windowMs: 10 * 60 * 1000,
  max: 20,
  message: { message: "Too many attempts. Try again later." },
});
app.use("/api/auth", authLimiter);

// --- Demo DB (بدليه لاحقًا بـ Mongo/Postgres) ---
const users = [];
const sanitizeText = (s) => xss((s || "").trim());

app.post("/api/auth/signup", async (req, res) => {
  try {
    let { fullName, email, password, role } = req.body;

    fullName = sanitizeText(fullName);
    email = (email || "").trim().toLowerCase();
    role = sanitizeText(role);

    if (!fullName || !email || !password)
      return res.status(400).json({ message: "Missing required fields." });

    if (!validator.isEmail(email))
      return res.status(400).json({ message: "Invalid email format." });

    const strong = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$/;
    if (!strong.test(password))
      return res.status(400).json({ message: "Weak password." });

    if (users.find(u => u.email === email))
      return res.status(409).json({ message: "Email already registered." });

    const passwordHash = await bcrypt.hash(password, 12);

    const user = {
      id: crypto.randomUUID(),
      fullName,
      email,
      role: role || "user",
      passwordHash,
      createdAt: new Date().toISOString()
    };
    users.push(user);

    return res.status(201).json({ message: "User created." });
  } catch (e) {
    console.error(e);
    res.status(500).json({ message: "Server error." });
  }
});

app.post("/api/auth/login", async (req, res) => {
  try {
    let { email, password } = req.body;
    email = (email || "").trim().toLowerCase();

    if (!email || !password)
      return res.status(400).json({ message: "Missing email or password." });

    const user = users.find(u => u.email === email);
    if (!user)
      return res.status(401).json({ message: "Invalid credentials." });

    const ok = await bcrypt.compare(password, user.passwordHash);
    if (!ok)
      return res.status(401).json({ message: "Invalid credentials." });

    const token = jwt.sign(
      { id: user.id, email: user.email, role: user.role },
      process.env.JWT_SECRET,
      { expiresIn: "2h" }
    );

    res.json({
      message: "Logged in.",
      token,
      user: { fullName: user.fullName, email: user.email, role: user.role }
    });
  } catch (e) {
    console.error(e);
    res.status(500).json({ message: "Server error." });
  }
});

app.get("/api/health", (req, res) => res.json({ ok: true }));

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log("Server running on", PORT));
