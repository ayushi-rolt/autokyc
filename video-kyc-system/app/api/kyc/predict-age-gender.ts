import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const fastApiUrl = process.env.NEXT_PUBLIC_FASTAPI_URL || "http://localhost:8000"

    console.log("[v0] Calling age/gender prediction")

    const response = await fetch(`${fastApiUrl}/predict-age-gender/`, {
      method: "POST",
      body: formData,
    })

    const data = await response.json()
    console.log("[v0] Age/Gender prediction result:", data)

    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    console.error("[v0] Age/Gender prediction error:", error)
    return NextResponse.json({ error: "Failed to predict age/gender", details: String(error) }, { status: 500 })
  }
}
