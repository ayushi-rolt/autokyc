import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const fastApiUrl = process.env.NEXT_PUBLIC_FASTAPI_URL || "http://localhost:8000"

    console.log("[v0] Calling document verification")

    const response = await fetch(`${fastApiUrl}/verify-document/`, {
      method: "POST",
      body: formData,
    })

    const data = await response.json()
    console.log("[v0] Document verification result:", data)

    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    console.error("[v0] Document verification error:", error)
    return NextResponse.json({ error: "Failed to verify document", details: String(error) }, { status: 500 })
  }
}
