import { type NextRequest, NextResponse } from "next/server"

const FASTAPI_URL = process.env.NEXT_PUBLIC_FASTAPI_URL || "http://localhost:8000"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()

    // Forward video blob and user_id to FastAPI
    const response = await fetch(`${FASTAPI_URL}/api/kyc/submit-video`, {
      method: "POST",
      body: formData,
    })

    if (!response.ok) {
      throw new Error(`FastAPI error: ${response.statusText}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Error submitting video:", error)
    return NextResponse.json({ error: "Failed to submit video" }, { status: 500 })
  }
}
