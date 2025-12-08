import { type NextRequest, NextResponse } from "next/server"

const FASTAPI_URL = process.env.NEXT_PUBLIC_FASTAPI_URL || "http://localhost:8000"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()

    console.log("[v0] API Route - Form data received")
    console.log("[v0] FASTAPI_URL:", FASTAPI_URL)
    console.log("[v0] Form keys:", Array.from(formData.keys()))

    const response = await fetch(`${FASTAPI_URL}/api/kyc/submit-user-data`, {
      method: "POST",
      body: formData,
    })

    console.log("[v0] FastAPI Response Status:", response.status)
    console.log("[v0] FastAPI Response OK:", response.ok)

    if (!response.ok) {
      const errorText = await response.text()
      console.log("[v0] FastAPI Error:", errorText)
      return NextResponse.json({ error: "Failed to submit user data", details: errorText }, { status: response.status })
    }

    const data = await response.json()
    console.log("[v0] FastAPI Success:", data)

    return NextResponse.json(data, {
      status: 200,
      headers: {
        "Content-Type": "application/json",
      },
    })
  } catch (error) {
    console.error("[v0] Error in API route:", error)
    return NextResponse.json({ error: "Failed to submit user data", details: String(error) }, { status: 500 })
  }
}
